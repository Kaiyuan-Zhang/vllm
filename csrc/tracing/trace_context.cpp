/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include "trace_context.h"
#include "trace_fifo.h"
#include <string.h>

static vllmTraceContextRing_t g_vllm_trace_ring = {
    .active_idx = 0xFFFFFFFFU,
    .capacity = VLLM_TRACE_CONTEXT_RING_CAPACITY,
    .version = 0,
    .slots = {}
};

extern "C" {

vllmTraceContextRing_t* vllm_trace_context_ring(void) {
    return &g_vllm_trace_ring;
}

void vllm_trace_context_update(
    uint64_t trace_id_hi,
    uint64_t trace_id_lo,
    uint64_t parent_span_id,
    uint64_t step_id,
    uint8_t trace_flags
) {
    // 1. Update legacy ring buffer
    uint32_t curr = __atomic_load_n(&g_vllm_trace_ring.active_idx, __ATOMIC_RELAXED);
    uint32_t next = (curr == 0xFFFFFFFFU) ? 0 : ((curr + 1) % VLLM_TRACE_CONTEXT_RING_CAPACITY);
    vllmTraceContext_t* slot = &g_vllm_trace_ring.slots[next];

    slot->trace_id_hi = trace_id_hi;
    slot->trace_id_lo = trace_id_lo;
    slot->parent_span_id = parent_span_id;
    slot->step_id = step_id;
    slot->trace_flags = trace_flags;
    memset(slot->_reserved, 0, sizeof(slot->_reserved));
    __atomic_store_n(&slot->is_valid, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&g_vllm_trace_ring.version, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&g_vllm_trace_ring.active_idx, next, __ATOMIC_RELEASE);

    // 2. Enqueue into GPU trace FIFO (marks DATA_READY, advances write_head)
    vllm_trace_fifo_enqueue(trace_id_hi, trace_id_lo, parent_span_id, step_id, trace_flags);
}

void vllm_trace_context_clear(void) {
    uint32_t curr = __atomic_load_n(&g_vllm_trace_ring.active_idx, __ATOMIC_RELAXED);
    if (curr != 0xFFFFFFFFU && curr < VLLM_TRACE_CONTEXT_RING_CAPACITY) {
        __atomic_store_n(&g_vllm_trace_ring.slots[curr].is_valid, 0, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&g_vllm_trace_ring.active_idx, 0xFFFFFFFFU, __ATOMIC_RELEASE);
    // Note: Do not call vllm_trace_fifo_reset() here to avoid destroying in-flight FIFO slots.
}

void vllm_trace_context_retire(uint64_t step_id) {
    // 1. Retire FIFO slot for this step
    vllm_trace_fifo_retire(step_id);

    // 2. Invalidate matching slot in legacy ring buffer
    for (uint32_t i = 0; i < VLLM_TRACE_CONTEXT_RING_CAPACITY; i++) {
        if (g_vllm_trace_ring.slots[i].step_id == step_id) {
            __atomic_store_n(&g_vllm_trace_ring.slots[i].is_valid, 0, __ATOMIC_RELEASE);
            break;
        }
    }
}

int vllm_trace_context_get_active(vllmTraceContext_t* out_ctx) {
    if (out_ctx == nullptr) {
        return 0;
    }

    // 1. Check GPU FIFO for hardware-active context
    vllmFifoSlot_t fifo_slot;
    if (vllm_trace_fifo_get_active(&fifo_slot)) {
        out_ctx->trace_id_hi = fifo_slot.trace_id_hi;
        out_ctx->trace_id_lo = fifo_slot.trace_id_lo;
        out_ctx->parent_span_id = fifo_slot.parent_span_id;
        out_ctx->step_id = fifo_slot.step_id;
        out_ctx->trace_flags = fifo_slot.trace_flags;
        out_ctx->is_valid = 1;
        memset(out_ctx->_reserved, 0, sizeof(out_ctx->_reserved));
        return 1;
    }

    // 2. Fall back to legacy ring buffer
    uint32_t idx = __atomic_load_n(&g_vllm_trace_ring.active_idx, __ATOMIC_ACQUIRE);
    if (idx != 0xFFFFFFFFU && idx < g_vllm_trace_ring.capacity) {
        vllmTraceContext_t* slot = &g_vllm_trace_ring.slots[idx];
        if (__atomic_load_n(&slot->is_valid, __ATOMIC_ACQUIRE)) {
            *out_ctx = *slot;
            return 1;
        }
    }

    return 0;
}

int vllm_trace_context_find_by_timestamp(uint64_t ptimer, vllmTraceContext_t* out_ctx) {
    if (out_ctx == nullptr || ptimer == 0) {
        return 0;
    }

    // 1. Try matching GPU hardware timestamp in FIFO
    vllmFifoSlot_t fifo_slot;
    if (vllm_trace_fifo_find_by_timestamp(ptimer, &fifo_slot)) {
        out_ctx->trace_id_hi = fifo_slot.trace_id_hi;
        out_ctx->trace_id_lo = fifo_slot.trace_id_lo;
        out_ctx->parent_span_id = fifo_slot.parent_span_id;
        out_ctx->step_id = fifo_slot.step_id;
        out_ctx->trace_flags = fifo_slot.trace_flags;
        out_ctx->is_valid = 1;
        memset(out_ctx->_reserved, 0, sizeof(out_ctx->_reserved));
        return 1;
    }

    // 2. Strict fail-closed: do not guess or fall back to active context when querying by timestamp
    return 0;
}

}  // extern "C"
