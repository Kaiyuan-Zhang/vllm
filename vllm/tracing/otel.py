# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import functools
import inspect
import os
import secrets
import traceback
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from typing import Any

from vllm.logger import init_logger
from vllm.tracing.utils import TRACE_HEADERS, LoadingSpanAttributes

logger = init_logger(__name__)

try:
    from opentelemetry import context, trace
    from opentelemetry.context.context import Context
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPGrpcExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPHttpExporter,
    )
    from opentelemetry.propagate import inject
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.id_generator import IdGenerator
    from opentelemetry.trace import (
        SpanKind,  # noqa: F401
        Tracer,
        set_tracer_provider,
    )
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _IS_OTEL_AVAILABLE = True
    otel_import_error_traceback = None


    class SecretsIdGenerator(IdGenerator):
        """An ID generator that draws from Python's `secrets` module (CSPRNG)
        instead of the standard library `random` module. This prevents span ID
        and trace ID collisions across multi-process tensor-parallel workers
        where Python's `random.seed()` is deterministically synchronized.
        """

        def generate_span_id(self) -> int:
            span_id = secrets.randbits(64)
            while span_id == trace.INVALID_SPAN_ID:
                span_id = secrets.randbits(64)
            return span_id

        def generate_trace_id(self) -> int:
            trace_id = secrets.randbits(128)
            while trace_id == trace.INVALID_TRACE_ID:
                trace_id = secrets.randbits(128)
            return trace_id

        def is_trace_id_random(self) -> bool:
            return True
except ImportError:
    _IS_OTEL_AVAILABLE = False
    otel_import_error_traceback = traceback.format_exc()
    trace = None  # type: ignore
    Context = Any  # type: ignore
    Tracer = Any  # type: ignore
    inject = None  # type: ignore
    Resource = None  # type: ignore
    SpanKind = Any  # type: ignore

_GLOBAL_TRACER_PROVIDER: Any = None


def is_otel_available() -> bool:
    return _IS_OTEL_AVAILABLE


def _get_tracer(name: str = __name__) -> Tracer:
    if _GLOBAL_TRACER_PROVIDER is not None:
        return _GLOBAL_TRACER_PROVIDER.get_tracer(name)
    return trace.get_tracer(name)


def init_otel_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
) -> Tracer:
    """Initializes the OpenTelemetry tracer provider."""
    if not _IS_OTEL_AVAILABLE:
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )

    # Store the endpoint in environment so child processes can inherit it
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = otlp_traces_endpoint

    resource_attrs = {}
    resource_attrs["service.name"] = instrumenting_module_name
    resource_attrs["vllm.instrumenting_module_name"] = instrumenting_module_name
    resource_attrs["vllm.process_id"] = str(os.getpid())
    if extra_attributes:
        resource_attrs.update(extra_attributes)
    resource = Resource.create(resource_attrs)

    trace_provider = TracerProvider(
        resource=resource,
        id_generator=SecretsIdGenerator(),
    )
    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    with suppress(Exception):
        set_tracer_provider(trace_provider)
    global _GLOBAL_TRACER_PROVIDER
    _GLOBAL_TRACER_PROVIDER = trace_provider

    atexit.register(trace_provider.shutdown)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter(endpoint: str):
    # Normalize endpoint: strip grpc:// prefix if present for OTLPGrpcExporter
    clean_endpoint = endpoint
    if clean_endpoint.startswith("grpc://"):
        clean_endpoint = clean_endpoint[len("grpc://") :]

    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        exporter = OTLPGrpcExporter(endpoint=clean_endpoint, insecure=True)
    elif protocol == "http/protobuf":
        exporter = OTLPHttpExporter(endpoint=clean_endpoint)
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")
    return exporter


_CURRENT_PROCESS_NAME: str | None = None


def init_otel_worker_tracer(
    instrumenting_module_name: str,
    process_kind: str,
    process_name: str,
) -> Tracer:
    """Backend-specific initialization for OpenTelemetry in a worker process."""
    global _CURRENT_PROCESS_NAME
    _CURRENT_PROCESS_NAME = process_name
    # Initialize the tracer if an OTLP endpoint is configured.
    # The endpoint is propagated via environment variable from the main process.
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not otlp_endpoint:
        return None

    extra_attrs = {
        "vllm.process_kind": process_kind,
        "vllm.process_name": process_name,
    }

    return init_otel_tracer(instrumenting_module_name, otlp_endpoint, extra_attrs)


def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None:
    """Extracts context from HTTP headers."""
    if _IS_OTEL_AVAILABLE and headers:
        return TraceContextTextMapPropagator().extract(headers)
    return None


def instrument_otel(func, span_name, attributes, record_exception):
    """Internal wrapper logic for sync and async functions."""
    # Pre-calculate static code attributes once (these don't change)
    code_attrs = {
        LoadingSpanAttributes.CODE_FUNCTION: func.__qualname__,
        LoadingSpanAttributes.CODE_NAMESPACE: func.__module__,
        LoadingSpanAttributes.CODE_FILEPATH: func.__code__.co_filename,
        LoadingSpanAttributes.CODE_LINENO: str(func.__code__.co_firstlineno),
    }
    if attributes:
        code_attrs.update(attributes)

    final_span_name = span_name or func.__qualname__
    module_name = func.__module__

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tracer = _get_tracer(module_name)
        ctx = _get_smart_context()
        with (
            tracer.start_as_current_span(
                final_span_name,
                context=ctx,
                attributes=code_attrs,
                record_exception=record_exception,
            ),
            propagate_trace_to_env(),
        ):
            return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        tracer = _get_tracer(module_name)
        ctx = _get_smart_context()
        with (
            tracer.start_as_current_span(
                final_span_name,
                context=ctx,
                attributes=code_attrs,
                record_exception=record_exception,
            ),
            propagate_trace_to_env(),
        ):
            return func(*args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


def manual_instrument_otel(
    span_name: str,
    start_time: int,
    end_time: int | None = None,
    attributes: dict[str, Any] | None = None,
    context: Context | None = None,
    kind: Any = None,  # SpanKind, but typed as Any for when OTEL unavailable
    links: list[Any] | None = None,
):
    """Manually create and end a span with explicit timestamps."""
    if not _IS_OTEL_AVAILABLE:
        return

    tracer = _get_tracer(__name__)
    # Use provided context, or fall back to smart context detection
    ctx = context if context is not None else _get_smart_context()

    span_kwargs: dict[str, Any] = {
        "name": span_name,
        "context": ctx,
        "start_time": start_time,
    }
    if kind is not None:
        span_kwargs["kind"] = kind
    if links:
        valid_links = [link for link in links if link is not None]
        if valid_links:
            span_kwargs["links"] = valid_links

    span = tracer.start_span(**span_kwargs)
    if attributes:
        span.set_attributes(attributes)
    if end_time is not None:
        span.end(end_time=end_time)
    else:
        span.end()


def create_trace_link_otel(trace_headers: dict[str, str] | None) -> Any:
    """Create an OpenTelemetry Link object from trace headers carrier dict."""
    if not _IS_OTEL_AVAILABLE or not trace_headers:
        return None

    ctx = extract_trace_context(trace_headers)
    if ctx is None:
        return None

    span_ctx = trace.get_current_span(ctx).get_span_context()
    if not span_ctx.is_valid:
        return None

    return trace.Link(span_ctx)


def start_request_span_otel(
    span_name: str,
    start_time: int,
    attributes: dict[str, Any] | None = None,
    context: Context | None = None,
    kind: Any = None,
) -> tuple[Any, dict[str, str] | None]:
    """Start an OpenTelemetry span and inject its context into a carrier dict.

    Returns:
        (span, trace_headers): The active span object and a dict of W3C
        trace headers (e.g. {'traceparent': ...}) representing this span's
        context for propagation to downstream workers/subsystems.

    """
    if not _IS_OTEL_AVAILABLE:
        return None, None

    tracer = _get_tracer(__name__)
    ctx = context if context is not None else _get_smart_context()

    span_kwargs: dict[str, Any] = {
        "name": span_name,
        "context": ctx,
        "start_time": start_time,
    }
    if kind is not None:
        span_kwargs["kind"] = kind

    span = tracer.start_span(**span_kwargs)
    if attributes:
        span.set_attributes(attributes)

    # Inject the new span's context into W3C trace headers
    carrier: dict[str, str] = {}
    span_ctx = trace.set_span_in_context(span)
    TraceContextTextMapPropagator().inject(carrier, context=span_ctx)

    return span, carrier


def start_step_span_otel(
    trace_headers: Mapping[str, Mapping[str, str]]
    | list[Mapping[str, str]]
    | None = None,
    step_id: int | None = None,
    num_tokens: int | None = None,
    attributes: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str] | None]:
    """Start a parent span in the scheduler representing a scheduled execution step.

    Creates 'vllm.scheduler.step' linked to all request traces in the scheduled batch,
    and returns the span along with a single W3C trace context carrier
    ({"traceparent": ...}) to propagate to workers over IPC (~55 bytes instead of
    full request headers).

    Args:
        trace_headers: Map of {req_id: carrier_dict} or list of carrier dicts
            from scheduled requests.
        step_id: Engine step counter / sequence.
        num_tokens: Total scheduled tokens for this step.
        attributes: Additional span attributes.

    Returns:
        tuple (span, carrier): The open step span object (to be ended when step
        completes in update_from_output) and a single carrier dict to serialize
        in SchedulerOutput.trace_headers.

    """
    if not _IS_OTEL_AVAILABLE:
        return None, None

    headers_list: list[Mapping[str, str]] = []
    request_ids: list[str] = []
    if isinstance(trace_headers, Mapping):
        for req_id, th in trace_headers.items():
            if th:
                headers_list.append(th)
                request_ids.append(str(req_id))
    elif isinstance(trace_headers, list):
        headers_list = [th for th in trace_headers if th]

    if not headers_list:
        return None, None

    tracer = _get_tracer(__name__)
    span_kwargs: dict[str, Any] = {
        "name": "vllm.scheduler.step",
    }

    if len(headers_list) == 1:
        parent_ctx = extract_trace_context(headers_list[0])
        if parent_ctx is not None:
            span_kwargs["context"] = parent_ctx
    else:
        links = []
        for th in headers_list:
            link = create_trace_link_otel(th)
            if link is not None:
                links.append(link)
        if links:
            span_kwargs["links"] = links

    span = tracer.start_span(**span_kwargs)

    span_attrs: dict[str, Any] = {
        "vllm.batch_size": len(headers_list),
    }
    if num_tokens is not None:
        span_attrs["vllm.num_tokens"] = num_tokens
    if step_id is not None:
        span_attrs["vllm.step_id"] = step_id
    if request_ids:
        span_attrs["vllm.request_ids"] = ",".join(request_ids)
    if attributes:
        span_attrs.update(attributes)
    if span_attrs:
        span.set_attributes(span_attrs)

    carrier: dict[str, str] = {}
    span_ctx = trace.set_span_in_context(span)
    TraceContextTextMapPropagator().inject(carrier, context=span_ctx)

    return span, carrier


@contextmanager
def trace_model_forward_otel(
    trace_headers: Mapping[str, Mapping[str, str]]
    | Mapping[str, str]
    | list[Mapping[str, str]]
    | None = None,
    attributes: dict[str, Any] | None = None,
    num_tokens: int | None = None,
    step_id: int | None = None,
    is_dummy: bool = False,
    defer_end: bool = False,
):
    """Context manager for tracing model forward passes.

    Creates a 'vllm.model.forward' span representing forward pass execution.
    Supports:
      1. Single W3C carrier (Mapping[str, str], e.g. {"traceparent": ...}):
         Extracted as parent context (typically from 'vllm.scheduler.step'),
         unifying multi-worker (TP/PP) ranks under the exact same trace ID.
      2. Dummy runs (is_dummy=True):
         Enqueues a dummy slot (trace_id=0, parent_span_id=0, step_id=step_id)
         into the GPU trace FIFO so GPU collectives executed during DP/EP
         dummy batches advance commit_head and do not miscorrelate to previous
         real steps.
      3. Legacy multi-request mapping ({req_id: carrier}):
         Parents single request or creates OpenTelemetry Links for batch.

    Updates the process-level C trace context ring buffer and lock-free GPU FIFO
    for external telemetry plugins (e.g. NCCL, DeepEP). If defer_end is True,
    context is detached upon exit but span.end() and FIFO slot retirement to
    VLLM_SLOT_COMPLETED are deferred until the returned ForwardTraceHandle.end()
    is called (e.g. at copy_event.synchronize).
    """
    if is_dummy:
        from vllm.tracing.trace_context import (
            ForwardTraceHandle,
            update_trace_context,
        )

        update_trace_context(
            trace_id=0,
            parent_span_id=0,
            step_id=step_id or 0,
            trace_flags=0,
        )
        handle = ForwardTraceHandle(
            span=None,
            token=None,
            step_id=step_id or 0,
        )
        try:
            yield handle
        finally:
            if not defer_end:
                handle.end()
        return

    if not _IS_OTEL_AVAILABLE:
        yield None
        return

    current_span = trace.get_current_span()
    if (
        current_span is not None
        and getattr(current_span, "name", None) == "vllm.model.forward"
    ):
        yield getattr(current_span, "_forward_trace_handle", None)
        return

    parent_ctx: Context | None = None
    links: list[Any] = []
    request_ids: list[str] = []
    batch_size = 1

    if isinstance(trace_headers, Mapping):
        # Check if trace_headers is a single W3C carrier dict (e.g. from scheduler)
        if (
            "traceparent" in trace_headers
            or "TRACEPARENT" in trace_headers
            or (
                trace_headers
                and all(
                    isinstance(v, (str, bytes)) for v in trace_headers.values()
                )
            )
        ):
            parent_ctx = extract_trace_context(trace_headers)
        else:
            # Legacy multi-request mapping: {req_id: carrier_dict}
            headers_list = []
            for req_id, th in trace_headers.items():
                if th:
                    headers_list.append(th)
                    request_ids.append(str(req_id))
            batch_size = len(headers_list)
            if len(headers_list) == 1:
                parent_ctx = extract_trace_context(headers_list[0])
            elif len(headers_list) > 1:
                for th in headers_list:
                    link = create_trace_link_otel(th)
                    if link is not None:
                        links.append(link)
    elif isinstance(trace_headers, list):
        headers_list = [th for th in trace_headers if th]
        batch_size = len(headers_list)
        if len(headers_list) == 1:
            parent_ctx = extract_trace_context(headers_list[0])
        elif len(headers_list) > 1:
            for th in headers_list:
                link = create_trace_link_otel(th)
                if link is not None:
                    links.append(link)

    if parent_ctx is None and not links:
        yield None
        return

    tracer = _get_tracer(__name__)
    span_kwargs: dict[str, Any] = {
        "name": "vllm.model.forward",
    }
    if parent_ctx is not None:
        span_kwargs["context"] = parent_ctx
    if links:
        span_kwargs["links"] = links

    span = tracer.start_span(**span_kwargs)
    span_attrs: dict[str, Any] = {}
    if num_tokens is not None:
        span_attrs["vllm.num_tokens"] = num_tokens
    if batch_size > 0:
        span_attrs["vllm.batch_size"] = batch_size
    if request_ids:
        span_attrs["vllm.request_ids"] = ",".join(request_ids)
    if step_id is not None:
        span_attrs["vllm.step_id"] = step_id
    if _CURRENT_PROCESS_NAME:
        span_attrs["vllm.process_name"] = _CURRENT_PROCESS_NAME
        if _CURRENT_PROCESS_NAME.startswith("Worker_"):
            with suppress(ValueError, IndexError):
                span_attrs["vllm.rank"] = int(_CURRENT_PROCESS_NAME.split("_")[-1])
    if attributes:
        span_attrs.update(attributes)
    if span_attrs:
        span.set_attributes(span_attrs)

    from vllm.tracing.trace_context import (
        ForwardTraceHandle,
        update_trace_context_from_span,
    )

    otel_ctx = trace.set_span_in_context(span)
    token = context.attach(otel_ctx)

    update_trace_context_from_span(span, step_id=step_id or 0)

    handle = ForwardTraceHandle(
        span=span,
        token=token,
        step_id=step_id or 0,
    )
    span._forward_trace_handle = handle

    try:
        yield handle
    except Exception:
        handle.end()
        raise
    finally:
        if not defer_end:
            handle.end()
        else:
            handle.detach_context()


def _get_smart_context() -> Context | None:
    """Determines the parent context.
    1. If a Span is already active in this process, use it.
    2. If not, extract from os.environ, handling the case-sensitivity mismatch.
    """
    current_span = trace.get_current_span()
    if current_span.get_span_context().is_valid:
        return None

    carrier = {}

    if tp := os.environ.get("traceparent", os.environ.get("TRACEPARENT")):  # noqa: SIM112
        carrier["traceparent"] = tp

    if ts := os.environ.get("tracestate", os.environ.get("TRACESTATE")):  # noqa: SIM112
        carrier["tracestate"] = ts

    if not carrier:
        carrier = dict(os.environ)

    return TraceContextTextMapPropagator().extract(carrier)


@contextmanager
def propagate_trace_to_env():
    """Temporarily injects the current OTel context into os.environ.
    This ensures that any subprocesses (like vLLM workers) spawned
    within this context inherit the correct traceparent.
    """
    if not _IS_OTEL_AVAILABLE:
        yield
        return

    # Capture original state of relevant keys
    original_state = {k: os.environ.get(k) for k in TRACE_HEADERS}

    try:
        # inject() writes 'traceparent' and 'tracestate' to os.environ
        inject(os.environ)
        yield

    finally:
        # Restore original environment
        for key, original_value in original_state.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
