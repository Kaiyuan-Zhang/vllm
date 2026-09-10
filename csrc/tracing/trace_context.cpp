/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include "trace_context.h"
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
    uint32_t curr = __atomic_load_n(&g_vllm_trace_ring.active_idx, __ATOMIC_RELAXED);
    uint32_t next = (curr == 0xFFFFFFFFU) ? 0 : ((curr + 1) % VLLM_TRACE_CONTEXT_RING_CAPACITY);

    vllmTraceContext_t* slot = &g_vllm_trace_ring.slots[next];
    __atomic_store_n(&slot->is_valid, 0, __ATOMIC_RELAXED);

    slot->trace_id_hi = trace_id_hi;
    slot->trace_id_lo = trace_id_lo;
    slot->parent_span_id = parent_span_id;
    slot->step_id = step_id;
    slot->trace_flags = trace_flags;
    memset(slot->_reserved, 0, sizeof(slot->_reserved));

    __atomic_store_n(&slot->is_valid, 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&g_vllm_trace_ring.version, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&g_vllm_trace_ring.active_idx, next, __ATOMIC_RELEASE);
}

void vllm_trace_context_clear(void) {
    uint32_t curr = __atomic_load_n(&g_vllm_trace_ring.active_idx, __ATOMIC_RELAXED);
    if (curr != 0xFFFFFFFFU && curr < VLLM_TRACE_CONTEXT_RING_CAPACITY) {
        __atomic_store_n(&g_vllm_trace_ring.slots[curr].is_valid, 0, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&g_vllm_trace_ring.active_idx, 0xFFFFFFFFU, __ATOMIC_RELEASE);
}

int vllm_trace_context_get_active(vllmTraceContext_t* out_ctx) {
    if (out_ctx == nullptr) {
        return 0;
    }
    uint32_t idx = __atomic_load_n(&g_vllm_trace_ring.active_idx, __ATOMIC_ACQUIRE);
    if (idx == 0xFFFFFFFFU || idx >= g_vllm_trace_ring.capacity) {
        return 0;
    }
    vllmTraceContext_t* slot = &g_vllm_trace_ring.slots[idx];
    if (!__atomic_load_n(&slot->is_valid, __ATOMIC_ACQUIRE)) {
        return 0;
    }
    *out_ctx = *slot;
    return 1;
}

}  // extern "C"
