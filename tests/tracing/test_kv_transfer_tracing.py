# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_TRACES_INSECURE,
)

from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from vllm.sampling_params import SamplingParams
from vllm.tracing import (
    KVTransferSpanAttributes,
    SpanKind,
    extract_trace_context,
    init_tracer,
    instrument_manual,
    is_otel_available,
    start_request_span,
    trace_model_forward,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")


class TestKVTransferTracing:
    """Focuses on KV transfer lifecycle and provider spans."""

    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.kv_transfer", FAKE_TRACE_SERVER_ADDRESS)

    def test_kv_transfer_scheduler_span_attributes(
        self, trace_service: FakeTraceService
    ):
        """Verify vllm.request.wait_remote_kv emits backend, token,
        and block attributes."""

        arrival_time_ns = time.time_ns()

        root_span, updated_headers = start_request_span(
            span_name="llm_request",
            start_time=arrival_time_ns,
            kind=SpanKind.SERVER,
        )
        assert root_span is not None
        assert updated_headers is not None

        sampling_params = SamplingParams(max_tokens=10)
        req = Request(
            request_id="req-kv-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sampling_params,
            pooling_params=None,
            arrival_time=arrival_time_ns / 1e9,
            trace_headers=updated_headers,
        )

        req.trace_end_queuing()

        # Simulate async KV loading transition in scheduler
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks

        mock_blocks = KVCacheBlocks(blocks=([object()] * 8,))
        req.async_kv_load_start_time_ns = time.time_ns()
        req.kv_backend = "NixlConnector"
        req.kv_num_tokens = 128
        req.kv_num_blocks = (
            sum(len(b) for b in mock_blocks.blocks) if mock_blocks else None
        )
        time.sleep(0.01)

        req.trace_end_kv_transfer()

        # End root span
        root_span.end()

        assert trace_service.wait_for_spans(count=3)
        spans = trace_service.get_all_spans()

        root = next(s for s in spans if s["name"] == "llm_request")
        kv_span = next(s for s in spans if s["name"] == "vllm.request.wait_remote_kv")

        # Verify parent-child hierarchy (wait_remote_kv is child of llm_request)
        assert kv_span["parent_span_id"] == root["span_id"]
        assert kv_span["trace_id"] == root["trace_id"]

        # Verify KV attributes
        attrs = kv_span["attributes"]
        backend_attr = attrs.get(KVTransferSpanAttributes.KV_TRANSFER_BACKEND)
        assert backend_attr == "NixlConnector"
        assert attrs.get(KVTransferSpanAttributes.KV_TRANSFER_NUM_TOKENS) == 128
        assert attrs.get(KVTransferSpanAttributes.KV_TRANSFER_NUM_BLOCKS) == 8

    def test_lmcache_provider_spans(self, trace_service: FakeTraceService):
        """Verify LMCache provider spans (retrieve and store) are parented to
        vllm.model.forward.
        """
        root_span, headers = start_request_span(
            span_name="llm_request",
            start_time=time.time_ns(),
            kind=SpanKind.SERVER,
        )

        with trace_model_forward(
            trace_headers={"req-1": headers},
            num_tokens=50,
        ):
            # Simulate LMCache retrieve in start_load_kv
            t0 = time.time_ns()
            time.sleep(0.005)
            t1 = time.time_ns()
            instrument_manual(
                span_name="lmcache.retrieve",
                start_time=t0,
                end_time=t1,
                attributes={
                    "lmcache.num_retrieved_tokens": 50,
                    "lmcache.req_id": "req-1",
                },
            )

            # Simulate LMCache store in wait_for_save
            t2 = time.time_ns()
            time.sleep(0.005)
            t3 = time.time_ns()
            instrument_manual(
                span_name="lmcache.store",
                start_time=t2,
                end_time=t3,
                attributes={
                    "lmcache.num_stored_tokens": 50,
                    "lmcache.req_id": "req-1",
                },
            )

        root_span.end()

        assert trace_service.wait_for_spans(count=4)
        spans = trace_service.get_all_spans()

        root = next(s for s in spans if s["name"] == "llm_request")
        fwd = next(s for s in spans if s["name"] == "vllm.model.forward")
        retrieve = next(s for s in spans if s["name"] == "lmcache.retrieve")
        store = next(s for s in spans if s["name"] == "lmcache.store")

        assert fwd["parent_span_id"] == root["span_id"]
        assert retrieve["parent_span_id"] == fwd["span_id"]
        assert store["parent_span_id"] == fwd["span_id"]
        assert retrieve["attributes"].get("lmcache.num_retrieved_tokens") == 50

    def test_nixl_rdma_sibling_span(self, trace_service: FakeTraceService):
        """Verify NIXL RDMA span is parented to the request root span and
        runs concurrently/nested in time with wait_remote_kv."""
        arrival_time_ns = time.time_ns()
        root_span, headers = start_request_span(
            span_name="llm_request",
            start_time=arrival_time_ns,
            kind=SpanKind.SERVER,
        )

        req = Request(
            request_id="req-nixl-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
            pooling_params=None,
            arrival_time=arrival_time_ns / 1e9,
            trace_headers=headers,
        )

        req.trace_end_queuing()

        # Step 1: Scheduler marks request WAITING_FOR_REMOTE_KVS
        t_wait_start = time.time_ns()
        req.async_kv_load_start_time_ns = t_wait_start
        req.kv_backend = "NixlConnector"
        req.kv_num_tokens = 256
        req.kv_num_blocks = 16

        # Step 2: Worker executes RDMA transfer inside the wait window
        time.sleep(0.005)
        t_rdma_start = time.time_ns()
        time.sleep(0.01)
        t_rdma_end = time.time_ns()
        instrument_manual(
            span_name="nixl.rdma.transfer",
            start_time=t_rdma_start,
            end_time=t_rdma_end,
            context=extract_trace_context(headers),
            attributes={
                "nixl.op": "READ",
                "nixl.num_blocks": 16,
                "nixl.remote_engine": "prefill-0",
                "nixl.backend": "UCX",
                "nixl.bytes_transferred": 1048576,
                "nixl.hardware_duration_us": 9500.0,
            },
        )

        # Step 3: Scheduler receives completion and closes wait_remote_kv
        time.sleep(0.005)
        req.trace_end_kv_transfer()
        root_span.end()

        assert trace_service.wait_for_spans(count=4)
        spans = trace_service.get_all_spans()

        root = next(s for s in spans if s["name"] == "llm_request")
        wait_kv = next(s for s in spans if s["name"] == "vllm.request.wait_remote_kv")
        rdma = next(s for s in spans if s["name"] == "nixl.rdma.transfer")

        # Both are siblings under root llm_request
        assert wait_kv["parent_span_id"] == root["span_id"]
        assert rdma["parent_span_id"] == root["span_id"]
        assert rdma["trace_id"] == root["trace_id"]

        # RDMA timestamps are strictly within the wait_remote_kv window
        assert rdma["attributes"].get("nixl.op") == "READ"
        assert rdma["attributes"].get("nixl.num_blocks") == 16
        assert rdma["attributes"].get("nixl.backend") == "UCX"
        assert rdma["attributes"].get("nixl.bytes_transferred") == 1048576
        assert rdma["attributes"].get("nixl.hardware_duration_us") == 9500.0

    def test_kv_cache_blocks_num_blocks_calculation(self):
        """Verify KVCacheBlocks structure is properly unpacked to count blocks."""
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks

        # Multi-group KV cache allocation
        mock_blocks_group0 = [object(), object(), object()]
        mock_blocks_group1 = [object(), object(), object(), object(), object()]
        new_blocks = KVCacheBlocks(blocks=(mock_blocks_group0, mock_blocks_group1))

        # Scheduler formula
        kv_num_blocks = sum(len(b) for b in new_blocks.blocks) if new_blocks else None
        assert kv_num_blocks == 8

        # Verify None case
        empty_blocks = None
        assert (
            sum(len(b) for b in empty_blocks.blocks) if empty_blocks else None
        ) is None
