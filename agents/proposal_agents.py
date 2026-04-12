from boardroom.schemas import Proposal
from env.schemas import EnvState
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent


class CFOProposalAgent(CFOAgent):
    def propose(self, state: EnvState) -> Proposal:
        action = self.act(state)

        return Proposal(
            agent="CFO",
            objective="Preserve runway and improve efficiency",
            actions=action,
            expected_impact="Lower burn, improved survival probability",
            risks=["Slower growth"],
            confidence=0.8,
        )


class CMOProposalAgent(CMOAgent):
    def propose(self, state: EnvState) -> Proposal:
        action = self.act(state)

        return Proposal(
            agent="CMO",
            objective="Maximize growth under CAC constraints",
            actions=action,
            expected_impact="Increased MRR growth",
            risks=["Higher CAC", "Burn risk"],
            confidence=0.75,
        )


class CPOProposalAgent(CPOAgent):
    def propose(self, state: EnvState) -> Proposal:
        action = self.act(state)

        return Proposal(
            agent="CPO",
            objective="Reduce churn and improve retention",
            actions=action,
            expected_impact="Higher NRR and lower churn",
            risks=["High R&D cost"],
            confidence=0.78,
        )
