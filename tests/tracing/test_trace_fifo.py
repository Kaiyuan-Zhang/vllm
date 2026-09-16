# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tracing import start_step_span, trace_model_forward
from vllm.tracing.trace_context import (
    clear_trace_context,
    find_trace_context_by_timestamp,
    find_trace_fifo_slot_by_timestamp,
    get_active_trace_fifo,
    get_trace_fifo,
    get_trace_fifo_ptr,
    is_trace_context_available,
    reset_trace_fifo,
    retire_trace_fifo,
    update_trace_context,
)


@pytest.mark.skipif(
    not is_trace_context_available(), reason="vllm_trace library not available"
)
class TestTraceFifo:
    def setup_method(self):
        clear_trace_context()
        reset_trace_fifo()

    def teardown_method(self):
        clear_trace_context()
        reset_trace_fifo()

    def test_fifo_symbols_and_structure(self):
        assert is_trace_context_available()
        fifo = get_trace_fifo()
        assert fifo is not None
        assert fifo.capacity == 2048

    def test_fifo_enqueue_and_retire(self):
        trace_id = 0x4BF92F3577B34DA6A3CE929D0E0E4736
        span_id = 0x00F067AA0BA902B7

        # Enqueue step 42
        update_trace_context(
            trace_id=trace_id,
            parent_span_id=span_id,
            step_id=42,
            trace_flags=1,
        )

        fifo = get_trace_fifo()
        assert fifo is not None
        assert fifo.write_head >= 1

        slot0 = fifo.slots[0]
        assert slot0.status == 1  # VLLM_SLOT_DATA_READY
        assert slot0.step_id == 42
        assert slot0.parent_span_id == span_id

        # Before GPU activation, get_active_trace_fifo should be None
        assert get_active_trace_fifo() is None

        # Simulate GPU activation (in real execution, done by
        # vllm_trace_fifo_activate kernel on stream)
        fifo_ptr = get_trace_fifo_ptr()
        assert fifo_ptr
        fifo_ptr.contents.slots[0].status = 2  # VLLM_SLOT_IN_USE
        fifo_ptr.contents.active_idx = 0
        fifo_ptr.contents.commit_head = 1

        # Now get_active_trace_fifo should return the active slot!
        active = get_active_trace_fifo()
        assert active is not None
        assert active.step_id == 42
        assert active.parent_span_id == span_id
        assert active.status == 2

        # Retire step 42 upon CPU copy_event sync -> slot remains COMPLETED
        retire_trace_fifo(42)
        assert fifo_ptr.contents.slots[0].status == 3  # VLLM_SLOT_COMPLETED

        # STRICT FAIL-CLOSED: once COMPLETED, get_active_trace_fifo must return None!
        assert get_active_trace_fifo() is None

    def test_fifo_timestamp_lookup(self):
        fifo_ptr = get_trace_fifo_ptr()
        assert fifo_ptr

        # Setup step 10: ts = 1000
        update_trace_context(
            trace_id=0x11111111111111112222222222222222,
            parent_span_id=0xAAAAAAAAAAAAAAAA,
            step_id=10,
            trace_flags=1,
        )
        fifo_ptr.contents.slots[0].status = 3  # VLLM_SLOT_COMPLETED
        fifo_ptr.contents.slots[0].gpu_start_ts = 1000
        fifo_ptr.contents.commit_head = 1

        # Setup step 11: ts = 2000
        update_trace_context(
            trace_id=0x33333333333333334444444444444444,
            parent_span_id=0xBBBBBBBBBBBBBBBB,
            step_id=11,
            trace_flags=1,
        )
        fifo_ptr.contents.slots[1].status = 2  # VLLM_SLOT_IN_USE
        fifo_ptr.contents.slots[1].gpu_start_ts = 2000
        fifo_ptr.contents.commit_head = 2

        # Timestamp before step 10 -> returns None
        assert find_trace_fifo_slot_by_timestamp(500) is None
        # find_trace_context_by_timestamp must also return None (no fallback!)
        assert find_trace_context_by_timestamp(500) is None

        # Timestamp during step 10 (1000 <= ts < 2000) -> returns step 10
        s10 = find_trace_fifo_slot_by_timestamp(1500)
        assert s10 is not None
        assert s10.step_id == 10
        assert s10.parent_span_id == 0xAAAAAAAAAAAAAAAA
        assert s10.gpu_start_ts == 1000

        ctx10 = find_trace_context_by_timestamp(1500)
        assert ctx10 is not None
        assert ctx10.step_id == 10
        assert ctx10.parent_span_id == 0xAAAAAAAAAAAAAAAA

        # Timestamp at or after step 11 (ts >= 2000) -> returns step 11
        s11 = find_trace_fifo_slot_by_timestamp(2500)
        assert s11 is not None
        assert s11.step_id == 11
        assert s11.parent_span_id == 0xBBBBBBBBBBBBBBBB
        assert s11.gpu_start_ts == 2000

        ctx11 = find_trace_context_by_timestamp(2500)
        assert ctx11 is not None
        assert ctx11.step_id == 11

    def test_fifo_wrap_around(self):
        fifo = get_trace_fifo()
        assert fifo is not None

        for step in range(2050):
            trace_id = 0x1000000000000000 + step
            span_id = 0x2000000000000000 + step
            update_trace_context(
                trace_id=trace_id,
                parent_span_id=span_id,
                step_id=step,
                trace_flags=1,
            )

        fifo = get_trace_fifo()
        assert fifo is not None
        assert fifo.write_head == 2050

        # Slot for step 2049 should be at index (2049 % 2048) = 1
        slot1 = fifo.slots[1]
        assert slot1.step_id == 2049

    def test_fifo_dummy_batch_execution(self):
        """Verify dummy runs enqueue slot with trace_id=0 and advance FIFO properly."""
        # Execute dummy model forward pass with is_dummy=True
        with trace_model_forward(is_dummy=True, step_id=50, defer_end=True) as handle:
            assert handle is not None
            assert handle.step_id == 50

            fifo = get_trace_fifo()
            assert fifo is not None
            assert fifo.write_head >= 1

            slot = fifo.slots[0]
            assert slot.status == 1  # VLLM_SLOT_DATA_READY
            assert slot.step_id == 50
            assert slot.trace_id_hi == 0
            assert slot.trace_id_lo == 0
            assert slot.parent_span_id == 0

            # Simulate GPU activation
            fifo_ptr = get_trace_fifo_ptr()
            assert fifo_ptr
            fifo_ptr.contents.slots[0].status = 2  # VLLM_SLOT_IN_USE
            fifo_ptr.contents.slots[0].gpu_start_ts = 5000
            fifo_ptr.contents.active_idx = 0
            fifo_ptr.contents.commit_head = 1

            # Timestamp lookup finds the dummy slot cleanly
            found = find_trace_fifo_slot_by_timestamp(5100)
            assert found is not None
            assert found.step_id == 50
            assert found.trace_id_hi == 0
            assert found.trace_id_lo == 0

            handle.end()

        # After retirement, status becomes COMPLETED
        assert fifo_ptr.contents.slots[0].status == 3  # VLLM_SLOT_COMPLETED
        assert get_active_trace_fifo() is None
        # Still discoverable via timestamp lookup
        found_completed = find_trace_fifo_slot_by_timestamp(5100)
        assert found_completed is not None
        assert found_completed.step_id == 50

    def test_step_span_carrier_and_forward(self):
        """Verify scheduler creates parent step span and worker parents
        under carrier.
        """
        trace_headers = {
            "req-1": {
                "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
            }
        }
        step_span, carrier = start_step_span(
            trace_headers=trace_headers,
            step_id=1,
            num_tokens=100,
        )
        assert carrier is not None
        assert "traceparent" in carrier
        assert carrier["traceparent"].startswith(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-"
        )

        # Worker creates forward span parented to carrier
        with trace_model_forward(
            trace_headers=carrier,
            step_id=1,
            num_tokens=100,
            defer_end=True,
        ) as forward_handle:
            assert forward_handle is not None
            assert forward_handle.span is not None

            # Trace ID must match the parent step span (and original request)
            span_ctx = forward_handle.span.get_span_context()
            assert f"{span_ctx.trace_id:032x}" == "4bf92f3577b34da6a3ce929d0e0e4736"

            forward_handle.end()

        if step_span is not None:
            step_span.end()
