import pytest

from vllm_hust_qos_scheduler import (
    QoSOrderingPolicy,
    QoSParams,
    QoSRuntimeState,
    RequestSnapshot,
)


def _request(request_id: str, deadline_ms: float) -> RequestSnapshot:
    state = QoSRuntimeState.from_params(
        QoSParams(ttft_slo_ms=deadline_ms),
        arrival_time=100.0,
        default_expected_output_tokens=16,
        wall_now=100.0,
        monotonic_now=50.0,
    )
    return RequestSnapshot(request_id, 100.0, 0, 8, 0, 0, 16, state)


def test_earlier_deadline_orders_first() -> None:
    policy = QoSOrderingPolicy()
    urgent = _request("urgent", 10)
    relaxed = _request("relaxed", 100)
    assert sorted([relaxed, urgent], key=policy.waiting_key)[0] is urgent


def test_runtime_state_uses_monotonic_deadline_and_counts_violation() -> None:
    state = _request("request", 10).qos_state
    assert state is not None
    assert state.next_token_deadline() == pytest.approx(50.01)
    assert state.observe_tokens(1, 50.02) == (1, 0)


def test_invalid_slo_fails_closed() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        QoSParams(ttft_slo_ms=0)
