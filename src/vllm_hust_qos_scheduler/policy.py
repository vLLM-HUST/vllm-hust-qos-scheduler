# SPDX-License-Identifier: Apache-2.0
"""Host-independent QoS deadline model and ordering from legacy PR #169."""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_EXPECTED_OUTPUT_TOKENS = 2**31 - 1


@dataclass(frozen=True)
class QoSParams:
    ttft_slo_ms: float | None = None
    tbt_slo_ms: float | None = None
    ttlt_slo_ms: float | None = None
    expected_output_tokens: int | None = None
    service_class: str | None = None

    def __post_init__(self) -> None:
        slos = (self.ttft_slo_ms, self.tbt_slo_ms, self.ttlt_slo_ms)
        if all(value is None for value in slos):
            raise ValueError("at least one QoS SLO must be specified")
        for value in slos:
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("SLO values must be finite and positive")
        if self.expected_output_tokens is not None and not (
            0 < self.expected_output_tokens <= MAX_EXPECTED_OUTPUT_TOKENS
        ):
            raise ValueError("expected_output_tokens is outside the supported range")
        if self.service_class is not None and not 1 <= len(self.service_class) <= 64:
            raise ValueError("service_class must contain 1 to 64 characters")


@dataclass
class QoSRuntimeState:
    ttft_deadline: float | None
    tbt_slo_s: float | None
    ttlt_deadline: float | None
    expected_output_tokens: int
    service_class: str | None
    first_token_time: float | None = None
    last_token_time: float | None = None
    ttft_observed: bool = False
    ttlt_observed: bool = False

    @classmethod
    def from_params(
        cls,
        params: QoSParams,
        *,
        arrival_time: float,
        default_expected_output_tokens: int,
        wall_now: float,
        monotonic_now: float,
    ) -> QoSRuntimeState:
        if not all(
            math.isfinite(value) for value in (arrival_time, wall_now, monotonic_now)
        ):
            raise ValueError("QoS timestamps must be finite")
        frontend_age_s = max(0.0, wall_now - arrival_time)

        def deadline(slo_ms: float | None) -> float | None:
            if slo_ms is None:
                return None
            return monotonic_now + slo_ms / 1000.0 - frontend_age_s

        return cls(
            ttft_deadline=deadline(params.ttft_slo_ms),
            tbt_slo_s=None if params.tbt_slo_ms is None else params.tbt_slo_ms / 1000,
            ttlt_deadline=deadline(params.ttlt_slo_ms),
            expected_output_tokens=(
                params.expected_output_tokens
                if params.expected_output_tokens is not None
                else default_expected_output_tokens
            ),
            service_class=params.service_class,
        )

    def next_token_deadline(self) -> float:
        deadlines: list[float] = []
        if self.first_token_time is None and self.ttft_deadline is not None:
            deadlines.append(self.ttft_deadline)
        elif self.tbt_slo_s is not None and self.last_token_time is not None:
            deadlines.append(self.last_token_time + self.tbt_slo_s)
        if self.ttlt_deadline is not None:
            deadlines.append(self.ttlt_deadline)
        return min(deadlines, default=math.inf)

    def observe_tokens(self, num_new_tokens: int, now: float) -> tuple[int, int]:
        if num_new_tokens <= 0:
            return 0, 0
        ttft_violation = 0
        tbt_violation = 0
        if not self.ttft_observed:
            ttft_violation = int(
                self.ttft_deadline is not None and now > self.ttft_deadline
            )
            self.ttft_observed = True
            self.first_token_time = now
        elif (
            self.tbt_slo_s is not None
            and self.last_token_time is not None
            and now - self.last_token_time > self.tbt_slo_s
        ):
            tbt_violation = 1
        self.last_token_time = now
        return ttft_violation, tbt_violation


@dataclass(frozen=True)
class RequestSnapshot:
    request_id: str
    arrival_time: float
    priority: int
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    max_tokens: int
    qos_state: QoSRuntimeState | None


@dataclass(frozen=True)
class QoSOrderingPolicy:
    hybrid_alpha_s_per_token: float = 0.0
    base_priority_enabled: bool = False

    def _remaining_work(self, request: RequestSnapshot) -> int:
        expected = (
            request.qos_state.expected_output_tokens
            if request.qos_state is not None
            else request.max_tokens
        )
        return max(0, request.num_prompt_tokens - request.num_computed_tokens) + max(
            0, expected - request.num_output_tokens
        )

    def _score(self, request: RequestSnapshot) -> float:
        if request.qos_state is None:
            return math.inf
        deadline = request.qos_state.next_token_deadline()
        if not math.isfinite(deadline):
            return math.inf
        return deadline + self.hybrid_alpha_s_per_token * self._remaining_work(request)

    def waiting_key(self, request: RequestSnapshot) -> tuple[object, ...]:
        priority = request.priority if self.base_priority_enabled else 0
        active = request.qos_state is not None and math.isfinite(
            request.qos_state.next_token_deadline()
        )
        return (
            priority,
            0 if active else 1,
            self._score(request) if active else 0.0,
            request.arrival_time if active or self.base_priority_enabled else 0.0,
            request.request_id if active or self.base_priority_enabled else "",
        )

    def running_key(self, request: RequestSnapshot) -> tuple[object, ...]:
        priority = request.priority if self.base_priority_enabled else 0
        active = request.qos_state is not None and math.isfinite(
            request.qos_state.next_token_deadline()
        )
        is_prefill = request.num_computed_tokens < request.num_prompt_tokens
        return (
            0 if active else 1,
            priority,
            self._score(request),
            int(is_prefill),
            request.arrival_time,
            request.request_id,
        )


__all__ = [
    "MAX_EXPECTED_OUTPUT_TOKENS",
    "QoSOrderingPolicy",
    "QoSParams",
    "QoSRuntimeState",
    "RequestSnapshot",
]
