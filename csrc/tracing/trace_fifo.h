/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#ifndef VLLM_TRACING_TRACE_FIFO_H_
#define VLLM_TRACING_TRACE_FIFO_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VLLM_TRACE_FIFO_CAPACITY 2048

/**
 * Slot status values representing the lifecycle of an inference forward pass.
 */
typedef enum {
    VLLM_SLOT_EMPTY      = 0,  // Free for CPU to write trace context
    VLLM_SLOT_DATA_READY = 1,  // CPU has written trace context; waiting for GPU
    VLLM_SLOT_IN_USE     = 2,  // GPU is actively executing this forward pass
    VLLM_SLOT_COMPLETED  = 3,  // Forward pass completed on GPU, retained for historical lookup
} vllmSlotStatus_t;

/**
 * Individual slot in the trace context FIFO.
 * Exactly 64 bytes (1 cache line) to eliminate false sharing across CPU cores.
 */
typedef struct {
    volatile uint32_t status;         // vllmSlotStatus_t (atomicCAS target)
    volatile uint32_t version;        // Seqlock version counter (even = stable, odd = writing)
    uint64_t trace_id_hi;             // Higher 64 bits of 128-bit trace ID (big-endian)
    uint64_t trace_id_lo;             // Lower 64 bits of 128-bit trace ID
    uint64_t parent_span_id;          // Span ID of vllm.model.forward
    uint64_t step_id;                 // Monotonic engine step counter
    volatile uint64_t gpu_start_ts;   // Hardware %globaltimer recorded by activate_kernel
    uint8_t  trace_flags;             // W3C trace flags (bit 0 = sampled)
    uint8_t  _pad[15];                // Padding to exactly 64 bytes
} __attribute__((aligned(64))) vllmFifoSlot_t;

/**
 * Lock-free circular FIFO stored in host-pinned mapped memory (cudaHostAllocMapped).
 * Accessible by both CPU and GPU via cache-coherent Unified Virtual Addressing (UVA).
 *
 * Header: 64 bytes.
 * Slots:  2048 slots * 64 bytes = 131,072 bytes (128 KB).
 * Total:  131,136 bytes (~128 KB host-pinned buffer for ~20s historical lookback).
 */
typedef struct {
    volatile uint64_t write_head;     // Advanced by CPU on enqueue
    volatile uint64_t commit_head;    // Advanced by GPU via atomicAdd_system on stream
    volatile uint32_t active_idx;     // Current active slot index (or 0xFFFFFFFF)
    uint32_t capacity;                // VLLM_TRACE_FIFO_CAPACITY (2048)
    uint8_t  _pad_header[40];         // Pad header to 64 bytes
    vllmFifoSlot_t slots[VLLM_TRACE_FIFO_CAPACITY];
} vllmTraceFifo_t;

/**
 * Returns a pointer to the singleton FIFO in host-pinned mapped memory.
 * Exported with default visibility for dlsym(RTLD_DEFAULT, "vllm_trace_fifo_global").
 */
__attribute__((visibility("default")))
vllmTraceFifo_t* vllm_trace_fifo_global(void);

/**
 * CPU producer: enqueues a new trace context into the next slot,
 * marks it DATA_READY, and advances write_head.
 */
__attribute__((visibility("default")))
void vllm_trace_fifo_enqueue(
    uint64_t trace_id_hi,
    uint64_t trace_id_lo,
    uint64_t parent_span_id,
    uint64_t step_id,
    uint8_t  trace_flags
);

/**
 * GPU activator: launches a 1-thread kernel onto stream that performs
 * atomicCAS_system(DATA_READY -> IN_USE). If DATA_READY, records hardware
 * %globaltimer, issues __threadfence_system(), updates active_idx, and
 * advances commit_head via atomicAdd_system. If not DATA_READY, does nothing.
 */
__attribute__((visibility("default")))
void vllm_trace_fifo_activate(void* stream);

/**
 * CPU consumer: called when copy_event.synchronize() unblocks on the host.
 * Marks the slot with step_id to VLLM_SLOT_COMPLETED, retaining it in the
 * circular buffer for historical timestamp lookups by external profilers.
 */
__attribute__((visibility("default")))
void vllm_trace_fifo_retire(uint64_t step_id);

/**
 * Profiler reader: safely reads the currently active slot using seqlock versioning.
 * Returns 1 if a slot currently in VLLM_SLOT_IN_USE was successfully read, 0 otherwise.
 */
__attribute__((visibility("default")))
int vllm_trace_fifo_get_active(vllmFifoSlot_t* out_slot);

/**
 * Looks up a trace context by GPU timestamp (%globaltimer).
 * Searches backwards from commit_head for the slot where slot.gpu_start_ts <= ptimer
 * within a 5-second plausibility bound, using seqlock versioning.
 * Returns 1 if a matching slot was found and safely copied, 0 otherwise.
 */
__attribute__((visibility("default")))
int vllm_trace_fifo_find_by_timestamp(uint64_t ptimer, vllmFifoSlot_t* out_slot);

/**
 * Resets all heads and slots in the FIFO to initial state.
 */
__attribute__((visibility("default")))
void vllm_trace_fifo_reset(void);

#ifdef __cplusplus
}
#endif

#endif  // VLLM_TRACING_TRACE_FIFO_H_
