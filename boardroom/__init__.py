__all__ = ["Boardroom", "Proposal", "NegotiationState"]


def __getattr__(name):
    if name == "Boardroom":
        from .boardroom import Boardroom

        return Boardroom
    if name in {"Proposal", "NegotiationState"}:
        from .schemas import NegotiationState, Proposal

        mapping = {
            "Proposal": Proposal,
            "NegotiationState": NegotiationState,
        }
        return mapping[name]
    raise AttributeError(f"module 'boardroom' has no attribute {name!r}")
