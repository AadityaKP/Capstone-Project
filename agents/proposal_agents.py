from boardroom.schemas import Proposal
from env.schemas import EnvState
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent


class CFOProposalAgent(CFOAgent):
    def __init__(self, llm_client=None, use_llm: bool = False):
        super().__init__()
        self.llm_client = llm_client
        self.use_llm = use_llm

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

        if self.use_llm and self.llm_client is not None:
            try:
                reasoning = self.llm_client.complete_text(
                    "You are the CFO of a SaaS startup. Be concise.",
                    f"Given this business state: MRR=${state.mrr:.0f}, "
                    f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
                    f"competitors={state.competitors}, "
                    f"consumer_confidence={state.consumer_confidence:.1f}. "
                    f"The proposed action is: {action}. "
                    f"In 2 sentences, explain the strategic rationale for this "
                    f"decision from the CFO's perspective."
                )
                if reasoning:
                    proposal = proposal.model_copy(
                        update={"expected_impact": reasoning}
                    )
            except Exception as e:
                print(f"[CFOProposalAgent] LLM refinement failed: {e}")

        return proposal


class CMOProposalAgent(CMOAgent):
    def __init__(self, llm_client=None, use_llm: bool = False):
        super().__init__()
        self.llm_client = llm_client
        self.use_llm = use_llm

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

        if self.use_llm and self.llm_client is not None:
            try:
                reasoning = self.llm_client.complete_text(
                    "You are the CMO of a SaaS startup. Be concise.",
                    f"Given this business state: MRR=${state.mrr:.0f}, "
                    f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
                    f"competitors={state.competitors}, "
                    f"consumer_confidence={state.consumer_confidence:.1f}. "
                    f"The proposed action is: {action}. "
                    f"In 2 sentences, explain the strategic rationale for this "
                    f"decision from the CMO's perspective."
                )
                if reasoning:
                    proposal = proposal.model_copy(
                        update={"expected_impact": reasoning}
                    )
            except Exception as e:
                print(f"[CMOProposalAgent] LLM refinement failed: {e}")

        return proposal


class CPOProposalAgent(CPOAgent):
    def __init__(self, llm_client=None, use_llm: bool = False):
        super().__init__()
        self.llm_client = llm_client
        self.use_llm = use_llm

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

        if self.use_llm and self.llm_client is not None:
            try:
                reasoning = self.llm_client.complete_text(
                    "You are the CPO of a SaaS startup. Be concise.",
                    f"Given this business state: MRR=${state.mrr:.0f}, "
                    f"cash=${state.cash:.0f}, churn={state.churn_smb:.2%}, "
                    f"competitors={state.competitors}, "
                    f"consumer_confidence={state.consumer_confidence:.1f}. "
                    f"The proposed action is: {action}. "
                    f"In 2 sentences, explain the strategic rationale for this "
                    f"decision from the CPO's perspective."
                )
                if reasoning:
                    proposal = proposal.model_copy(
                        update={"expected_impact": reasoning}
                    )
            except Exception as e:
                print(f"[CPOProposalAgent] LLM refinement failed: {e}")

        return proposal
