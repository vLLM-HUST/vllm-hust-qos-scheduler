"""QoS request and ordering policy with inert runtime metadata."""

from .policy import (
    MAX_EXPECTED_OUTPUT_TOKENS,
    QoSOrderingPolicy,
    QoSParams,
    QoSRuntimeState,
    RequestSnapshot,
)


class VllmHustQosSchedulerContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


__all__ = [
    "MAX_EXPECTED_OUTPUT_TOKENS",
    "QoSOrderingPolicy",
    "QoSParams",
    "QoSRuntimeState",
    "RequestSnapshot",
    "VllmHustQosSchedulerContractProposal",
]
