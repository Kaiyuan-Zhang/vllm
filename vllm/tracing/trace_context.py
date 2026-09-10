# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM Trace Context Exporter for in-process telemetry / profiler plugins.

Exposes an OpenTelemetry trace context ring buffer into host memory and exports
`vllm_trace_context_ring` as a global C symbol (`RTLD_GLOBAL`), allowing external
plugins such as NCCL profiler plugins and DeepEP to discover the active forward pass
trace context without compile-time coupling to vLLM.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import shutil
import subprocess
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

VLLM_TRACE_CONTEXT_RING_CAPACITY = 64


class VllmTraceContext(ctypes.Structure):
    _fields_ = [
        ("trace_id_hi", ctypes.c_uint64),
        ("trace_id_lo", ctypes.c_uint64),
        ("parent_span_id", ctypes.c_uint64),
        ("step_id", ctypes.c_uint64),
        ("trace_flags", ctypes.c_uint8),
        ("is_valid", ctypes.c_uint8),
        ("_reserved", ctypes.c_uint8 * 6),
    ]

    def to_dict(self) -> dict[str, Any]:
        trace_id = (self.trace_id_hi << 64) | self.trace_id_lo
        return {
            "trace_id": f"{trace_id:032x}",
            "parent_span_id": f"{self.parent_span_id:016x}",
            "step_id": self.step_id,
            "trace_flags": self.trace_flags,
            "is_valid": bool(self.is_valid),
        }


class VllmTraceContextRing(ctypes.Structure):
    _fields_ = [
        ("active_idx", ctypes.c_uint32),
        ("capacity", ctypes.c_uint32),
        ("version", ctypes.c_uint64),
        ("slots", VllmTraceContext * VLLM_TRACE_CONTEXT_RING_CAPACITY),
    ]


_trace_lib: ctypes.CDLL | None = None


def _find_or_build_trace_lib() -> ctypes.CDLL | None:
    global _trace_lib
    if _trace_lib is not None:
        return _trace_lib

    # Candidate paths for libvllm_trace.so
    current_dir = Path(__file__).parent.resolve()
    vllm_dir = current_dir.parent
    candidate_paths = [
        vllm_dir / "libvllm_trace.so",
        vllm_dir / "libs" / "libvllm_trace.so",
        current_dir / "libvllm_trace.so",
    ]

    for p in candidate_paths:
        if p.is_file():
            try:
                _trace_lib = ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                _setup_lib_signatures(_trace_lib)
                return _trace_lib
            except Exception as e:
                logger.warning("Failed to load trace context library %s: %s", p, e)

    # Check system library search
    sys_path = ctypes.util.find_library("vllm_trace")
    if sys_path:
        try:
            _trace_lib = ctypes.CDLL(sys_path, mode=ctypes.RTLD_GLOBAL)
            _setup_lib_signatures(_trace_lib)
            return _trace_lib
        except Exception as e:
            logger.warning("Failed to load trace context library from system: %s", e)

    # If running from source in development, attempt compiling
    # csrc/tracing/trace_context.cpp
    source_path = vllm_dir.parent / "csrc" / "tracing" / "trace_context.cpp"
    if source_path.is_file():
        compiler = shutil.which("g++") or shutil.which("gcc") or shutil.which("clang++")
        if compiler:
            target_so = vllm_dir / "libvllm_trace.so"
            try:
                cmd = [
                    compiler,
                    "-O3",
                    "-shared",
                    "-fPIC",
                    str(source_path),
                    "-o",
                    str(target_so),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                _trace_lib = ctypes.CDLL(str(target_so), mode=ctypes.RTLD_GLOBAL)
                _setup_lib_signatures(_trace_lib)
                return _trace_lib
            except Exception as e:
                logger.warning("Failed to compile %s: %s", source_path, e)

    return None


def _setup_lib_signatures(lib: ctypes.CDLL) -> None:
    lib.vllm_trace_context_ring.restype = ctypes.POINTER(VllmTraceContextRing)
    lib.vllm_trace_context_ring.argtypes = []

    lib.vllm_trace_context_update.restype = None
    lib.vllm_trace_context_update.argtypes = [
        ctypes.c_uint64,  # trace_id_hi
        ctypes.c_uint64,  # trace_id_lo
        ctypes.c_uint64,  # parent_span_id
        ctypes.c_uint64,  # step_id
        ctypes.c_uint8,  # trace_flags
    ]

    lib.vllm_trace_context_clear.restype = None
    lib.vllm_trace_context_clear.argtypes = []

    lib.vllm_trace_context_get_active.restype = ctypes.c_int
    lib.vllm_trace_context_get_active.argtypes = [ctypes.POINTER(VllmTraceContext)]


# Initialize library on import
_find_or_build_trace_lib()


def is_trace_context_available() -> bool:
    """Returns True if the C trace context library was successfully loaded."""
    return _trace_lib is not None


def get_trace_context_ring() -> VllmTraceContextRing | None:
    """Returns a copy of the singleton trace context ring buffer structure."""
    if _trace_lib is None:
        return None
    ptr = _trace_lib.vllm_trace_context_ring()
    if not ptr:
        return None
    return ptr.contents


def get_active_trace_context() -> VllmTraceContext | None:
    """Returns the currently active trace context slot, or None if invalid/empty."""
    if _trace_lib is None:
        return None
    ctx = VllmTraceContext()
    if _trace_lib.vllm_trace_context_get_active(ctypes.byref(ctx)):
        return ctx
    return None


def update_trace_context(
    trace_id: int,
    parent_span_id: int,
    step_id: int = 0,
    trace_flags: int = 1,
) -> None:
    """Updates the ring buffer with an active trace context."""
    if _trace_lib is None:
        return
    trace_id_hi = (trace_id >> 64) & 0xFFFFFFFFFFFFFFFF
    trace_id_lo = trace_id & 0xFFFFFFFFFFFFFFFF
    _trace_lib.vllm_trace_context_update(
        ctypes.c_uint64(trace_id_hi),
        ctypes.c_uint64(trace_id_lo),
        ctypes.c_uint64(parent_span_id),
        ctypes.c_uint64(step_id),
        ctypes.c_uint8(trace_flags),
    )


def update_trace_context_from_span(span: Any, step_id: int = 0) -> None:
    """Updates the ring buffer using an OpenTelemetry Span object."""
    if span is None or _trace_lib is None:
        return
    span_ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
    if span_ctx is None or not getattr(span_ctx, "is_valid", False):
        return
    flags = int(span_ctx.trace_flags) if hasattr(span_ctx, "trace_flags") else 1
    update_trace_context(
        trace_id=span_ctx.trace_id,
        parent_span_id=span_ctx.span_id,
        step_id=step_id,
        trace_flags=flags,
    )


def clear_trace_context() -> None:
    """Clears/invalidates the active trace context in the ring buffer."""
    if _trace_lib is None:
        return
    _trace_lib.vllm_trace_context_clear()


class ForwardTraceHandle:
    """Handle for an active model forward trace span.

    Allows detaching the OpenTelemetry thread context at the end of CPU forward
    enqueue while keeping the span open and the in-process C trace context ring
    valid until GPU execution completes (e.g. at copy_event.synchronize).
    """

    def __init__(
        self,
        span: Any = None,
        token: Any = None,
        step_id: int = 0,
    ) -> None:
        self.span = span
        self.token = token
        self.step_id = step_id
        self._ended = False

    def detach_context(self) -> None:
        """Detaches this span from the current thread context while keeping
        the span open."""
        if self.token is not None:
            with contextlib.suppress(Exception):
                from opentelemetry import context

                context.detach(self.token)
            self.token = None

    def end(self) -> None:
        """Ends the forward span and clears the C trace context ring active slot."""
        if self._ended:
            return
        self._ended = True
        self.detach_context()
        if self.span is not None:
            with contextlib.suppress(Exception):
                self.span.end()
        clear_trace_context()

    def __enter__(self) -> ForwardTraceHandle:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end()
