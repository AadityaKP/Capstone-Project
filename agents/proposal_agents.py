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


class _LLMRationaleSupport:
    def _init_llm_support(self, llm_client=None, use_llm: bool = False) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm
        self._rationale_cache: dict[tuple, str] = {}

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
    def __init__(self, llm_client=None, use_llm: bool = False, scale: float = 1.0):
        super().__init__(scale=scale)
        self._init_llm_support(llm_client=llm_client, use_llm=use_llm)

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

        reasoning = self._get_rationale(
            state,
            "You are the CFO of a SaaS startup. Be concise.",
            f"Given this business state: MRR=${state.mrr:.0f}, "
            f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
            f"competitors={state.competitors}, "
            f"consumer_confidence={state.consumer_confidence:.1f}. "
            f"The proposed action is: {action}. "
            f"In 2 sentences, explain the strategic rationale for this "
            f"decision from the CFO's perspective.",
        )
        if reasoning:
            proposal = proposal.model_copy(update={"rationale": reasoning})

        return proposal


class CMOProposalAgent(CMOAgent, _LLMRationaleSupport):
    def __init__(self, llm_client=None, use_llm: bool = False, scale: float = 1.0):
        super().__init__(scale=scale)
        self._init_llm_support(llm_client=llm_client, use_llm=use_llm)

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

        reasoning = self._get_rationale(
            state,
            "You are the CMO of a SaaS startup. Be concise.",
            f"Given this business state: MRR=${state.mrr:.0f}, "
            f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
            f"competitors={state.competitors}, "
            f"consumer_confidence={state.consumer_confidence:.1f}. "
            f"The proposed action is: {action}. "
            f"In 2 sentences, explain the strategic rationale for this "
            f"decision from the CMO's perspective.",
        )
        if reasoning:
            proposal = proposal.model_copy(update={"rationale": reasoning})

        return proposal


class CPOProposalAgent(CPOAgent, _LLMRationaleSupport):
    def __init__(self, llm_client=None, use_llm: bool = False, scale: float = 1.0):
        super().__init__(scale=scale)
        self._init_llm_support(llm_client=llm_client, use_llm=use_llm)

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

        reasoning = self._get_rationale(
            state,
            "You are the CPO of a SaaS startup. Be concise.",
            f"Given this business state: MRR=${state.mrr:.0f}, "
            f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
            f"competitors={state.competitors}, "
            f"consumer_confidence={state.consumer_confidence:.1f}. "
            f"The proposed action is: {action}. "
            f"In 2 sentences, explain the strategic rationale for this "
            f"decision from the CPO's perspective.",
        )
        if reasoning:
            proposal = proposal.model_copy(update={"rationale": reasoning})

        return proposal
