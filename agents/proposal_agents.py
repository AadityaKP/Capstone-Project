from copy import deepcopy
from typing import Optional

from boardroom.schemas import Proposal
from env.schemas import EnvState
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent


def _state_cache_key(state: EnvState) -> tuple:
    shock_label = getattr(state, "shock_label", None) or "NONE"
    return (
        round(state.mrr / 10_000),
        round(state.cash / 10_000),
        round(state.churn_smb / 0.05) * 0.05,
        round(state.consumer_confidence / 10),
        shock_label,
    )


# How far a lever is pulled back when last month's prediction for the role's
# own KPI missed on direction (plan section 3.2b). Toward hold, never past it:
# "the lever did not do what we said it would, so lean on it less until we
# know more" is the one response that is defensible for every lever.
TRACK_RECORD_DAMPING = 0.75


class _LLMRationaleSupport:
    def _init_llm_support(self, llm_client=None, use_llm: bool = False,
                          expectation=None) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm
        self._rationale_cache: dict[tuple, str] = {}
        # Product path only (plan section 3): the expected_delta predictor and
        # the agent's own track record. Both None on every research arm, which
        # keeps the heuristic action and the Proposal byte-identical there.
        self.expectation = expectation
        self.track_record: dict | None = None

    def set_track_record(self, record: dict | None) -> None:
        self.track_record = deepcopy(record) if record else None

    def _adapt_to_track_record(self, role: str, action: dict) -> tuple[dict, Optional[str]]:
        """Damp the role's lever when its last prediction missed on sign.

        Returns (action, sentence). The sentence is the agent saying, in one
        line, what it changed and why - the trace carries it as `adaptation`.
        """
        from boardroom.expectation import role_miss

        miss = role_miss(self.track_record, role)
        if miss is None:
            return action, None
        kpi_word = {"mrr_pct": "MRR", "cash_pct": "cash", "churn_pp": "churn"}[miss["kpi"]]
        unit = "pp" if miss["kpi"] == "churn_pp" else "%"
        because = (
            f"Last month's plan expected {kpi_word} {miss['expected']:+.1f}{unit} "
            f"and saw {miss['realized']:+.1f}{unit}"
        )
        adapted = deepcopy(action)
        if role == "CMO" and adapted.get("marketing", {}).get("spend", 0) > 0:
            adapted["marketing"]["spend"] = adapted["marketing"]["spend"] * TRACK_RECORD_DAMPING
            return adapted, f"{because}; marketing is held back {round((1 - TRACK_RECORD_DAMPING) * 100)}% this month."
        if role == "CPO" and adapted.get("product", {}).get("r_and_d_spend", 0) > 0:
            adapted["product"]["r_and_d_spend"] = adapted["product"]["r_and_d_spend"] * TRACK_RECORD_DAMPING
            return adapted, f"{because}; product spend is held back {round((1 - TRACK_RECORD_DAMPING) * 100)}% this month."
        if role == "CFO" and adapted.get("hiring", {}).get("hires", 0) > 0:
            adapted["hiring"]["hires"] = 0
            return adapted, f"{because}; hiring waits this month."
        return action, f"{because}; nothing to pull back on this month."

    def _finish_proposal(self, role: str, state: EnvState, action: dict,
                         proposal: Proposal) -> tuple[dict, Proposal]:
        """Apply the track record and attach expected_delta. No-op on the
        research path, where neither is set."""
        adaptation = None
        if self.track_record is not None:
            action, adaptation = self._adapt_to_track_record(role, action)
            proposal = proposal.model_copy(update={"actions": action, "adaptation": adaptation})
        if self.expectation is not None:
            proposal = proposal.model_copy(update={"expected_delta": self.expectation(state, action)})
        return action, proposal

    def _track_record_clause(self) -> str:
        if self.track_record is None:
            return ""
        from boardroom.expectation import describe_track_record

        return (
            " Your own track record from last month: "
            + describe_track_record(self.track_record).replace("\n", "; ")
            + ". If a lever did not do what you predicted, say what you are changing."
        )

    def _get_rationale(
        self,
        state: EnvState,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[str]:
        if not self.use_llm or self.llm_client is None:
            return None

        cache_key = _state_cache_key(state)
        cached = self._rationale_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            if hasattr(self.llm_client, "complete_text"):
                reasoning = self.llm_client.complete_text(system_prompt, user_prompt)
            elif hasattr(self.llm_client, "complete"):
                reasoning = self.llm_client.complete(system_prompt, user_prompt)
            else:
                reasoning = ""
        except Exception as exc:
            print(f"[{self.__class__.__name__}] LLM refinement failed: {exc}")
            return None

        if reasoning:
            self._rationale_cache[cache_key] = reasoning
            return reasoning

        return None


class CFOProposalAgent(CFOAgent, _LLMRationaleSupport):
    def __init__(self, llm_client=None, use_llm: bool = False, scale: float = 1.0,
                 corridor: str = "legacy", expectation=None):
        super().__init__(scale=scale, corridor=corridor)
        self._init_llm_support(llm_client=llm_client, use_llm=use_llm, expectation=expectation)

    def propose(self, state: EnvState) -> Proposal:
        action = self.act(state)

        proposal = Proposal(
            agent="CFO",
            objective="Preserve runway and improve efficiency",
            actions=action,
            expected_impact="Lower burn, improved survival probability",
            risks=["Slower growth"],
            confidence=0.8,
        )
        action, proposal = self._finish_proposal("CFO", state, action, proposal)

        reasoning = self._get_rationale(
            state,
            "You are the CFO of a SaaS startup. Be concise.",
            f"Given this business state: MRR=${state.mrr:.0f}, "
            f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
            f"competitors={state.competitors}, "
            f"consumer_confidence={state.consumer_confidence:.1f}. "
            f"The proposed action is: {action}. "
            f"In 2 sentences, explain the strategic rationale for this "
            f"decision from the CFO's perspective." + self._track_record_clause(),
        )
        if reasoning:
            proposal = proposal.model_copy(update={"rationale": reasoning})

        return proposal


class CMOProposalAgent(CMOAgent, _LLMRationaleSupport):
    def __init__(self, llm_client=None, use_llm: bool = False, scale: float = 1.0,
                 corridor: str = "legacy", expectation=None):
        super().__init__(scale=scale, corridor=corridor)
        self._init_llm_support(llm_client=llm_client, use_llm=use_llm, expectation=expectation)

    def propose(self, state: EnvState) -> Proposal:
        action = self.act(state)

        proposal = Proposal(
            agent="CMO",
            objective="Maximize growth under CAC constraints",
            actions=action,
            expected_impact="Increased MRR growth",
            risks=["Higher CAC", "Burn risk"],
            confidence=0.75,
        )
        action, proposal = self._finish_proposal("CMO", state, action, proposal)

        reasoning = self._get_rationale(
            state,
            "You are the CMO of a SaaS startup. Be concise.",
            f"Given this business state: MRR=${state.mrr:.0f}, "
            f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
            f"competitors={state.competitors}, "
            f"consumer_confidence={state.consumer_confidence:.1f}. "
            f"The proposed action is: {action}. "
            f"In 2 sentences, explain the strategic rationale for this "
            f"decision from the CMO's perspective." + self._track_record_clause(),
        )
        if reasoning:
            proposal = proposal.model_copy(update={"rationale": reasoning})

        return proposal


class CPOProposalAgent(CPOAgent, _LLMRationaleSupport):
    def __init__(self, llm_client=None, use_llm: bool = False, scale: float = 1.0,
                 corridor: str = "legacy", expectation=None):
        super().__init__(scale=scale, corridor=corridor)
        self._init_llm_support(llm_client=llm_client, use_llm=use_llm, expectation=expectation)

    def propose(self, state: EnvState) -> Proposal:
        action = self.act(state)

        proposal = Proposal(
            agent="CPO",
            objective="Reduce churn and improve retention",
            actions=action,
            expected_impact="Higher NRR and lower churn",
            risks=["High R&D cost"],
            confidence=0.78,
        )
        action, proposal = self._finish_proposal("CPO", state, action, proposal)

        reasoning = self._get_rationale(
            state,
            "You are the CPO of a SaaS startup. Be concise.",
            f"Given this business state: MRR=${state.mrr:.0f}, "
            f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
            f"competitors={state.competitors}, "
            f"consumer_confidence={state.consumer_confidence:.1f}. "
            f"The proposed action is: {action}. "
            f"In 2 sentences, explain the strategic rationale for this "
            f"decision from the CPO's perspective." + self._track_record_clause(),
        )
        if reasoning:
            proposal = proposal.model_copy(update={"rationale": reasoning})

        return proposal
