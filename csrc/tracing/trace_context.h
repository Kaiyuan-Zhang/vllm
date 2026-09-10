/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#ifndef VLLM_TRACING_TRACE_CONTEXT_H_
#define VLLM_TRACING_TRACE_CONTEXT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VLLM_TRACE_CONTEXT_RING_CAPACITY 64

/**
 * OpenTelemetry trace context for a single forward pass / batch execution.
 * Fixed-size layout compatible with C and C++.
 */
typedef struct {
    uint64_t trace_id_hi;      // Higher 64 bits of 128-bit trace ID (big-endian)
    uint64_t trace_id_lo;      // Lower 64 bits of 128-bit trace ID
    uint64_t parent_span_id;   // 64-bit span ID of vllm.model.forward
    uint64_t step_id;          // Monotonic engine step / iteration counter
    uint8_t  trace_flags;      // Trace flags (bit 0: sampled)
    uint8_t  is_valid;         // 1 if slot contains valid trace context, 0 otherwise
    uint8_t  _reserved[6];     // Padding to 8-byte alignment
} vllmTraceContext_t;

/**
 * Ring buffer storing recent trace contexts.
 * SWMR: Single-Writer (main engine / worker thread), Multi-Reader (NCCL / DeepEP plugins).
 */
typedef struct {
    uint32_t active_idx;       // Index of most recent slot in [0, capacity - 1], or 0xFFFFFFFF
    uint32_t capacity;         // VLLM_TRACE_CONTEXT_RING_CAPACITY (64)
    uint64_t version;          // Monotonically increasing write sequence counter
    vllmTraceContext_t slots[VLLM_TRACE_CONTEXT_RING_CAPACITY];
} vllmTraceContextRing_t;

/**
 * Returns a pointer to the singleton trace context ring buffer.
 *
 * Exported with default visibility so external plugins (NCCL profiler, DeepEP)
 * can resolve it dynamically via dlsym(RTLD_DEFAULT, "vllm_trace_context_ring").
 */
__attribute__((visibility("default")))
vllmTraceContextRing_t* vllm_trace_context_ring(void);

/**
 * Updates the ring buffer with a new active trace context.
 * Called by vLLM engine / model runner before forward execution.
 */
__attribute__((visibility("default")))
void vllm_trace_context_update(
    uint64_t trace_id_hi,
    uint64_t trace_id_lo,
    uint64_t parent_span_id,
    uint64_t step_id,
    uint8_t trace_flags
);

/**
 * Invalidate / clear the active trace context in the ring buffer.
 */
__attribute__((visibility("default")))
void vllm_trace_context_clear(void);

/**
 * Helper to copy the currently active trace context into out_ctx.
 * Returns 1 if an active valid context is present, 0 otherwise.
 */
__attribute__((visibility("default")))
int vllm_trace_context_get_active(vllmTraceContext_t* out_ctx);

#ifdef __cplusplus
}
#endif

#endif  // VLLM_TRACING_TRACE_CONTEXT_H_
