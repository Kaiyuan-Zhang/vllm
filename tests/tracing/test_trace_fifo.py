# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tracing.trace_context import (
    clear_trace_context,
    get_active_trace_fifo,
    get_trace_fifo,
    get_trace_fifo_ptr,
    is_trace_context_available,
    retire_trace_fifo,
    update_trace_context,
)


@pytest.mark.skipif(
    not is_trace_context_available(), reason="vllm_trace library not available"
)
class TestTraceFifo:
    def setup_method(self):
        clear_trace_context()

    def teardown_method(self):
        clear_trace_context()

    def test_fifo_symbols_and_structure(self):
        assert is_trace_context_available()
        fifo = get_trace_fifo()
        assert fifo is not None
        assert fifo.capacity == 64

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

        # Retire step 42 upon CPU copy_event sync
        retire_trace_fifo(42)
        assert fifo_ptr.contents.slots[0].status == 0  # VLLM_SLOT_EMPTY
        assert get_active_trace_fifo() is None

    def test_fifo_wrap_around(self):
        fifo = get_trace_fifo()
        assert fifo is not None

        for step in range(70):
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
        assert fifo.write_head == 70

        # Slot for step 69 should be at index (69 % 64) = 5
        slot5 = fifo.slots[5]
        assert slot5.step_id == 69
