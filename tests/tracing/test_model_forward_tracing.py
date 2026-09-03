# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_TRACES_INSECURE,
)

from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from vllm.tracing import (
    SpanKind,
    create_trace_link,
    init_tracer,
    instrument_manual,
    is_otel_available,
    start_request_span,
    trace_model_forward,
)

pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")


class TestModelForwardTracing:

    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.forward_pass", FAKE_TRACE_SERVER_ADDRESS)

    def test_create_trace_link(self):
        """Verify create_trace_link creates valid OTel Link or returns None."""
        carrier = {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        }
        link = create_trace_link(carrier)
        assert link is not None
        assert hasattr(link, "context")

        assert create_trace_link(None) is None
        assert create_trace_link({}) is None

    def test_trace_model_forward_single_request(
        self, trace_service: FakeTraceService
    ):
        """Verify vllm.model.forward parents to single request's root span and
        inner spans become children of vllm.model.forward."""
        root_span, headers = start_request_span(
            span_name="llm_request",
            start_time=time.time_ns(),
            kind=SpanKind.SERVER,
        )

        with trace_model_forward(
            trace_headers={"req-1": headers},
            num_tokens=100,
        ):
            # Inside forward pass, simulate an inner operation (e.g. attention kernel)
            instrument_manual(
                span_name="vllm.model.attention",
                start_time=time.time_ns(),
                end_time=time.time_ns(),
                attributes={
                    "layer": "0",
                },
            )

        root_span.end()

        assert trace_service.wait_for_spans(count=3)
        spans = trace_service.get_all_spans()

        root = next(s for s in spans if s["name"] == "llm_request")
        fwd = next(s for s in spans if s["name"] == "vllm.model.forward")
        inner = next(s for s in spans if s["name"] == "vllm.model.attention")

        # Verify hierarchy: root -> fwd -> inner
        assert fwd["parent_span_id"] == root["span_id"]
        assert fwd["trace_id"] == root["trace_id"]
        assert inner["parent_span_id"] == fwd["span_id"]
        assert inner["trace_id"] == root["trace_id"]
        assert fwd["attributes"].get("vllm.num_tokens") == 100
        assert fwd["attributes"].get("vllm.batch_size") == 1

    def test_trace_model_forward_multi_request(
        self, trace_service: FakeTraceService
    ):
        """Verify multi-request batch forward pass creates links to all requests."""
        root1, h1 = start_request_span("llm_request", time.time_ns())
        root2, h2 = start_request_span("llm_request", time.time_ns())

        with trace_model_forward(
            trace_headers={"req-1": h1, "req-2": h2},
            num_tokens=250,
        ):
            pass

        root1.end()
        root2.end()

        assert trace_service.wait_for_spans(count=3)
        spans = trace_service.get_all_spans()
        fwd = next(s for s in spans if s["name"] == "vllm.model.forward")

        assert fwd["attributes"].get("vllm.num_tokens") == 250
        assert fwd["attributes"].get("vllm.batch_size") == 2
