# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, NamedTuple, TypeAlias

# Import the implementation details
from .otel import (
    SpanKind,
    create_trace_link_otel,
    extract_trace_context,
    init_otel_tracer,
    init_otel_worker_tracer,
    instrument_otel,
    is_otel_available,
    manual_instrument_otel,
    otel_import_error_traceback,
    start_request_span_otel,
    trace_model_forward_otel,
)
from .utils import (
    SpanAttributes,
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)

__all__ = [
    "instrument",
    "instrument_manual",
    "init_tracer",
    "maybe_init_worker_tracer",
    "is_tracing_available",
    "SpanAttributes",
    "SpanKind",
    "extract_trace_context",
    "extract_trace_headers",
    "log_tracing_disabled_warning",
    "contains_trace_headers",
    "otel_import_error_traceback",
    "start_request_span",
    "create_trace_link",
    "trace_model_forward",
]

BackendAvailableFunc: TypeAlias = Callable[[], bool]

InstrumentFunc: TypeAlias = Callable[..., Any]
InstrumentManualFunc: TypeAlias = Callable[..., Any]
StartRequestSpanFunc: TypeAlias = Callable[..., Any]
InitTracerFunc: TypeAlias = Callable[..., Any]
InitWorkerTracerFunc: TypeAlias = Callable[..., Any]
CreateTraceLinkFunc: TypeAlias = Callable[..., Any]
TraceModelForwardFunc: TypeAlias = Callable[..., Any]


class TracingBackend(NamedTuple):
    is_available: BackendAvailableFunc
    init_tracer: InitTracerFunc
    init_worker_tracer: InitWorkerTracerFunc
    instrument: InstrumentFunc
    manual_instrument: InstrumentManualFunc
    start_request_span: StartRequestSpanFunc
    create_trace_link: CreateTraceLinkFunc
    trace_model_forward: TraceModelForwardFunc


_REGISTERED_TRACING_BACKENDS: dict[str, TracingBackend] = {
    "otel": TracingBackend(
        is_available=is_otel_available,
        init_tracer=init_otel_tracer,
        init_worker_tracer=init_otel_worker_tracer,
        instrument=instrument_otel,
        manual_instrument=manual_instrument_otel,
        start_request_span=start_request_span_otel,
        create_trace_link=create_trace_link_otel,
        trace_model_forward=trace_model_forward_otel,
    ),
}


def init_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
):
    backend = _REGISTERED_TRACING_BACKENDS.get("otel")
    if backend and backend.is_available():
        return backend.init_tracer(
            instrumenting_module_name, otlp_traces_endpoint, extra_attributes
        )


def maybe_init_worker_tracer(
    instrumenting_module_name: str,
    process_kind: str,
    process_name: str,
):
    backend = _REGISTERED_TRACING_BACKENDS.get("otel")
    if backend and backend.is_available():
        return backend.init_worker_tracer(
            instrumenting_module_name, process_kind, process_name
        )


def instrument(
    obj: Callable | None = None,
    *,
    span_name: str = "",
    attributes: dict[str, str] | None = None,
    record_exception: bool = True,
):
    """Generic decorator to instrument functions."""
    if obj is None:
        return functools.partial(
            instrument,
            span_name=span_name,
            attributes=attributes,
            record_exception=record_exception,
        )

    # Dispatch to OTel (and potentially others later)
    backend = _REGISTERED_TRACING_BACKENDS.get("otel")
    if backend and backend.is_available():
        return backend.instrument(
            func=obj,
            span_name=span_name,
            attributes=attributes,
            record_exception=record_exception,
        )
    else:
        return obj


def instrument_manual(
    span_name: str,
    start_time: int,
    end_time: int | None = None,
    attributes: dict[str, Any] | None = None,
    context: Any = None,
    kind: Any = None,
    links: list[Any] | None = None,
):
    """Manually create a span with explicit timestamps.

    Args:
        span_name: Name of the span to create.
        start_time: Start time in nanoseconds since epoch.
        end_time: Optional end time in nanoseconds. If None, ends immediately.
        attributes: Optional dict of span attributes.
        context: Optional trace context (e.g., from extract_trace_context).
        kind: Optional SpanKind (e.g., SpanKind.SERVER).
        links: Optional list of trace Link objects.

    """
    backend = _REGISTERED_TRACING_BACKENDS.get("otel")
    if backend and backend.is_available():
        return backend.manual_instrument(
            span_name, start_time, end_time, attributes, context, kind, links
        )
    else:
        return None


def start_request_span(
    span_name: str,
    start_time: int,
    attributes: dict[str, Any] | None = None,
    context: Any = None,
    kind: Any = None,
) -> tuple[Any, dict[str, str] | None]:
    """Start a span and return it along with injected W3C trace headers.

    Args:
        span_name: Name of the span to create.
        start_time: Start time in nanoseconds since epoch.
        attributes: Optional dict of span attributes.
        context: Optional trace context (e.g., from extract_trace_context).
        kind: Optional SpanKind (e.g., SpanKind.SERVER).

    """
    backend = _REGISTERED_TRACING_BACKENDS.get("otel")
    if backend and backend.is_available():
        return backend.start_request_span(
            span_name, start_time, attributes, context, kind
        )
    else:
        return None, None


def create_trace_link(trace_headers: dict[str, str] | None) -> Any:
    """Create an OpenTelemetry Link from W3C trace headers.

    Args:
        trace_headers: Carrier dict containing W3C trace context headers.

    Returns:
        Link object if tracing is available and headers are valid, else None.

    """
    backend = _REGISTERED_TRACING_BACKENDS.get("otel")
    if backend and backend.is_available():
        return backend.create_trace_link(trace_headers)
    else:
        return None


@contextmanager
def trace_model_forward(
    trace_headers: Any = None,
    attributes: dict[str, Any] | None = None,
    num_tokens: int | None = None,
):
    """Context manager for tracing model forward passes.

    Creates a 'vllm.model.forward' span representing forward pass execution,
    sets it as current span so inner operations (KV transfer, kernels) are
    properly parented, and links it to the active requests.
    """
    backend = _REGISTERED_TRACING_BACKENDS.get("otel")
    if backend and backend.is_available():
        with backend.trace_model_forward(trace_headers, attributes, num_tokens):
            yield
    else:
        yield


def is_tracing_available() -> bool:
    """Returns True if any tracing backend (OTel, Profiler, etc.) is available.
    Use this to guard expensive tracing logic in the main code.
    """
    check_available = [
        backend.is_available() for backend in _REGISTERED_TRACING_BACKENDS.values()
    ]
    return any(check_available)
