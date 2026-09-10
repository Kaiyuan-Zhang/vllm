/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include "trace_fifo.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <atomic>

static vllmTraceFifo_t* g_trace_fifo = nullptr;

#ifdef __CUDACC__
__global__ void vllm_trace_fifo_activate_kernel(vllmTraceFifo_t* dev_fifo) {
    uint64_t commit = dev_fifo->commit_head;
    uint32_t slot_idx = (uint32_t)(commit % dev_fifo->capacity);
    vllmFifoSlot_t* slot = &dev_fifo->slots[slot_idx];

    // Non-blocking CAS on host-pinned mapped memory over PCIe / NVLink
    unsigned int old_status = atomicCAS_system(
        (unsigned int*)&slot->status,
        (unsigned int)VLLM_SLOT_DATA_READY,
        (unsigned int)VLLM_SLOT_IN_USE
    );

    if (old_status == (unsigned int)VLLM_SLOT_DATA_READY) {
        dev_fifo->active_idx = slot_idx;
        atomicAdd_system((unsigned long long*)&dev_fifo->commit_head, 1ULL);
    }
}
#endif

extern "C" {

__attribute__((visibility("default")))
vllmTraceFifo_t* vllm_trace_fifo_global(void) {
    if (!g_trace_fifo) {
#ifdef __CUDACC__
        cudaError_t err = cudaHostAlloc((void**)&g_trace_fifo, sizeof(vllmTraceFifo_t), cudaHostAllocMapped);
        if (err != cudaSuccess || !g_trace_fifo) {
            g_trace_fifo = (vllmTraceFifo_t*)calloc(1, sizeof(vllmTraceFifo_t));
        }
#else
        g_trace_fifo = (vllmTraceFifo_t*)calloc(1, sizeof(vllmTraceFifo_t));
#endif
        if (g_trace_fifo) {
            memset(g_trace_fifo, 0, sizeof(vllmTraceFifo_t));
            g_trace_fifo->capacity = VLLM_TRACE_FIFO_CAPACITY;
            g_trace_fifo->active_idx = 0xFFFFFFFFU;
        }
    }
    return g_trace_fifo;
}

__attribute__((visibility("default")))
void vllm_trace_fifo_enqueue(
    uint64_t trace_id_hi,
    uint64_t trace_id_lo,
    uint64_t parent_span_id,
    uint64_t step_id,
    uint8_t  trace_flags
) {
    vllmTraceFifo_t* fifo = vllm_trace_fifo_global();
    if (!fifo) return;

    uint64_t head = fifo->write_head;
    uint32_t slot_idx = (uint32_t)(head % fifo->capacity);
    vllmFifoSlot_t* slot = &fifo->slots[slot_idx];

    // Seqlock: odd version indicates write in progress
    slot->version = (slot->version + 1) | 1;
    std::atomic_thread_fence(std::memory_order_release);

    slot->trace_id_hi = trace_id_hi;
    slot->trace_id_lo = trace_id_lo;
    slot->parent_span_id = parent_span_id;
    slot->step_id = step_id;
    slot->trace_flags = trace_flags;

    std::atomic_thread_fence(std::memory_order_release);
    // Seqlock: even version indicates write complete
    slot->version = slot->version + 1;

    std::atomic_thread_fence(std::memory_order_release);
    slot->status = VLLM_SLOT_DATA_READY;

    std::atomic_thread_fence(std::memory_order_seq_cst);
    fifo->write_head = head + 1;
}

__attribute__((visibility("default")))
void vllm_trace_fifo_activate(void* stream) {
    vllmTraceFifo_t* fifo = vllm_trace_fifo_global();
    if (!fifo) return;
#ifdef __CUDACC__
    cudaStream_t s = (cudaStream_t)stream;
    vllm_trace_fifo_activate_kernel<<<1, 1, 0, s>>>(fifo);
#else
    // CPU fallback if compiled without CUDA: activate directly
    uint64_t commit = fifo->commit_head;
    uint32_t slot_idx = (uint32_t)(commit % fifo->capacity);
    vllmFifoSlot_t* slot = &fifo->slots[slot_idx];
    if (slot->status == VLLM_SLOT_DATA_READY) {
        slot->status = VLLM_SLOT_IN_USE;
        fifo->active_idx = slot_idx;
        fifo->commit_head = commit + 1;
    }
#endif
}

__attribute__((visibility("default")))
void vllm_trace_fifo_retire(uint64_t step_id) {
    vllmTraceFifo_t* fifo = vllm_trace_fifo_global();
    if (!fifo) return;
    for (uint32_t i = 0; i < fifo->capacity; i++) {
        if (fifo->slots[i].step_id == step_id && fifo->slots[i].status == VLLM_SLOT_IN_USE) {
            fifo->slots[i].status = VLLM_SLOT_EMPTY;
            break;
        }
    }
}

__attribute__((visibility("default")))
int vllm_trace_fifo_get_active(vllmFifoSlot_t* out_slot) {
    vllmTraceFifo_t* fifo = vllm_trace_fifo_global();
    if (!fifo || !out_slot) return 0;
    uint32_t idx = fifo->active_idx;
    if (idx >= fifo->capacity) return 0;
    vllmFifoSlot_t* slot = &fifo->slots[idx];
    if (slot->status != VLLM_SLOT_IN_USE) return 0;

    for (int retry = 0; retry < 3; retry++) {
        uint32_t v1 = slot->version;
        if (v1 & 1) continue;
        std::atomic_thread_fence(std::memory_order_acquire);

        *out_slot = *slot;

        std::atomic_thread_fence(std::memory_order_acquire);
        uint32_t v2 = slot->version;
        if (v1 == v2 && out_slot->status == VLLM_SLOT_IN_USE) {
            return 1;
        }
    }
    return 0;
}

__attribute__((visibility("default")))
void vllm_trace_fifo_reset(void) {
    vllmTraceFifo_t* fifo = vllm_trace_fifo_global();
    if (!fifo) return;
    fifo->write_head = 0;
    fifo->commit_head = 0;
    fifo->active_idx = 0xFFFFFFFFU;
    for (uint32_t i = 0; i < fifo->capacity; i++) {
        fifo->slots[i].status = VLLM_SLOT_EMPTY;
    }
}

}  // extern "C"
