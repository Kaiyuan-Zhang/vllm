# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import time

import pytest
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_TRACES_INSECURE,
)

from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from vllm.tracing import (
    SpanKind,
    init_tracer,
    is_otel_available,
    start_request_span,
    trace_model_forward,
)
from vllm.tracing.trace_context import (
    VllmTraceContextRing,
    clear_trace_context,
    get_active_trace_context,
    get_trace_context_ring,
    is_trace_context_available,
    update_trace_context,
)

pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")


class TestTraceContextRing:
    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.trace_context_ring", FAKE_TRACE_SERVER_ADDRESS)
        clear_trace_context()
        yield
        clear_trace_context()

    def test_c_symbol_resolution_via_dlsym(self):
        """Verify external profilers (NCCL/DeepEP) can resolve vllm_trace_context_ring
        using RTLD_DEFAULT / dlopen(NULL) without compile-time dependencies."""
        assert is_trace_context_available()
        handle = ctypes.CDLL(None)
        assert hasattr(handle, "vllm_trace_context_ring")

        func = handle.vllm_trace_context_ring
        func.restype = ctypes.POINTER(VllmTraceContextRing)
        func.argtypes = []

        ring_ptr = func()
        assert bool(ring_ptr)
        ring = ring_ptr.contents
        assert ring.capacity == 64

    def test_ring_update_and_clear(self):
        """Verify updating and clearing trace context in the ring buffer."""
        trace_id = 0x123456789ABCDEF0123456789ABCDEF0
        parent_span_id = 0xFEDCBA9876543210
        step_id = 101

        update_trace_context(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            step_id=step_id,
            trace_flags=1,
        )

        active = get_active_trace_context()
        assert active is not None
        d = active.to_dict()
        assert d["trace_id"] == f"{trace_id:032x}"
        assert d["parent_span_id"] == f"{parent_span_id:016x}"
        assert d["step_id"] == 101
        assert d["trace_flags"] == 1
        assert d["is_valid"] is True

        ring = get_trace_context_ring()
        assert ring is not None
        assert ring.version >= 1

        clear_trace_context()
        assert get_active_trace_context() is None

    def test_ring_wrap_around(self):
        """Verify monotonic versioning and wrap-around modulo capacity (64)."""
        for i in range(70):
            update_trace_context(
                trace_id=i + 1,
                parent_span_id=i + 100,
                step_id=i,
            )

        ring = get_trace_context_ring()
        assert ring is not None
        assert ring.version >= 70
        assert ring.active_idx == (70 - 1) % 64
        active = get_active_trace_context()
        assert active is not None
        assert active.step_id == 69

    def test_trace_model_forward_deferred_end(self, trace_service: FakeTraceService):
        """Verify trace_model_forward with defer_end=True keeps span open and ring
        slot active until ForwardTraceHandle.end() is called."""
        root_span, headers = start_request_span(
            span_name="llm_request",
            start_time=time.time_ns(),
            kind=SpanKind.SERVER,
        )

        trace_handle = None
        with trace_model_forward(
            trace_headers={"req-1": headers},
            num_tokens=50,
            step_id=123,
            defer_end=True,
        ) as handle:
            trace_handle = handle
            assert handle is not None
            # In-process C ring buffer has active valid slot
            active = get_active_trace_context()
            assert active is not None
            assert active.step_id == 123
            assert active.is_valid == 1

        # Outside the with-block: span should STILL NOT have ended!
        assert not trace_handle._ended
        active_after_exit = get_active_trace_context()
        assert active_after_exit is not None
        assert active_after_exit.step_id == 123

        # Now simulate copy_event.synchronize() completion
        trace_handle.end()
        assert trace_handle._ended
        assert get_active_trace_context() is None

        root_span.end()
        assert trace_service.wait_for_spans(count=2)
        spans = trace_service.get_all_spans()
        fwd = next(s for s in spans if s["name"] == "vllm.model.forward")
        assert fwd["attributes"].get("vllm.step_id") == 123
        assert fwd["attributes"].get("vllm.num_tokens") == 50
