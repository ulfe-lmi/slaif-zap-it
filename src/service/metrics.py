"""Process-local, finite-cardinality Prometheus metrics for the loopback API."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from prometheus_client.registry import CollectorRegistry

__all__ = ["CONTENT_TYPE_LATEST", "ServiceMetrics"]


class ServiceMetrics:
    """A custom registry with no default process/runtime collectors."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests = Counter(
            "zap_it_requests_total",
            "Requests by stable outcome code.",
            ("outcome",),
            registry=self.registry,
        )
        self.completions = Counter(
            "zap_it_completions_total",
            "Successful completions by bounded response dimensions.",
            ("verbosity", "response_format"),
            registry=self.registry,
        )
        self.busy = Counter("zap_it_busy_total", "Busy admissions.", registry=self.registry)
        self.not_ready = Counter(
            "zap_it_not_ready_total", "Not-ready responses.", registry=self.registry
        )
        self.timeout = Counter("zap_it_timeout_total", "Timeout responses.", registry=self.registry)
        self.cancelled = Counter(
            "zap_it_cancelled_total", "Cancellation responses.", registry=self.registry
        )
        self.response_limits = Counter(
            "zap_it_response_limit_total",
            "Response/resource-limit failures.",
            registry=self.registry,
        )
        self.active = Gauge(
            "zap_it_active_inference", "Currently active inference calls.", registry=self.registry
        )
        self.readiness = Gauge(
            "zap_it_readiness", "Readiness verdict, one when ready.", registry=self.registry
        )
        buckets = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)
        self.request_duration = Histogram(
            "zap_it_request_duration_seconds",
            "Request duration.",
            buckets=buckets,
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "zap_it_inference_duration_seconds",
            "Inference duration.",
            buckets=buckets,
            registry=self.registry,
        )
        self.serialization_duration = Histogram(
            "zap_it_serialization_duration_seconds",
            "Serialization duration.",
            buckets=buckets,
            registry=self.registry,
        )
        self.response_bytes = Histogram(
            "zap_it_response_bytes",
            "Successful response byte sizes.",
            buckets=(1024, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 256_000_000),
            registry=self.registry,
        )
        self.object_count = Histogram(
            "zap_it_object_count",
            "Successful object counts.",
            buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256),
            registry=self.registry,
        )
        self.artifact_count = Histogram(
            "zap_it_artifact_count",
            "Successful artifact counts.",
            buckets=(0, 1, 2, 4, 8, 16, 32, 48, 64),
            registry=self.registry,
        )
        self.gpu_allocated = Gauge(
            "zap_it_torch_gpu_allocated_bytes",
            "Logical cuda:0 allocated bytes.",
            registry=self.registry,
        )
        self.gpu_reserved = Gauge(
            "zap_it_torch_gpu_reserved_bytes",
            "Logical cuda:0 reserved bytes.",
            registry=self.registry,
        )
        self.model_initializations = Counter(
            "zap_it_model_initializations_total",
            "Pinned model-holder initialization outcomes.",
            ("component", "outcome"),
            registry=self.registry,
        )
        self.residency_transitions = Counter(
            "zap_it_residency_transitions_total",
            "Bounded model residency transition outcomes.",
            ("direction", "outcome"),
            registry=self.registry,
        )
        self.residency_transition_duration = Histogram(
            "zap_it_residency_transition_duration_seconds",
            "Model residency transition duration.",
            ("direction",),
            buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
            registry=self.registry,
        )

    def observe_error(self, code: str) -> None:
        self.requests.labels(outcome=code).inc()
        if code == "service_busy":
            self.busy.inc()
        elif code == "not_ready":
            self.not_ready.inc()
        elif code == "timeout":
            self.timeout.inc()
        elif code == "cancelled":
            self.cancelled.inc()
        elif code in {"response_too_large", "insufficient_memory", "insufficient_shm"}:
            self.response_limits.inc()

    def observe_success(
        self,
        *,
        verbosity: int,
        response_format: str,
        response_bytes: int,
        objects: int,
        artifacts: int,
    ) -> None:
        self.requests.labels(outcome="success").inc()
        self.completions.labels(verbosity=str(verbosity), response_format=response_format).inc()
        self.response_bytes.observe(response_bytes)
        self.object_count.observe(objects)
        self.artifact_count.observe(artifacts)

    def update_gpu(self) -> None:
        """Sample only logical ``cuda:0`` counters when Torch is available."""
        try:
            import torch

            if torch.cuda.is_available():
                self.gpu_allocated.set(float(torch.cuda.memory_allocated(0)))
                self.gpu_reserved.set(float(torch.cuda.memory_reserved(0)))
        except (ImportError, RuntimeError, AttributeError, TypeError):
            # CPU installs and transient CUDA teardown do not make metrics
            # collection a request failure.
            return

    def observe_model_initialization(self, component: str, outcome: str) -> None:
        """Record only fixed operator component/outcome labels."""
        if component not in {"sam2", "clip", "blip3", "registry"}:
            component = "registry"
        if outcome not in {"success", "failure"}:
            outcome = "failure"
        self.model_initializations.labels(component=component, outcome=outcome).inc()

    def observe_residency_transition(
        self, direction: str, outcome: str, duration_seconds: float
    ) -> None:
        """Record fixed-label swap/restore observations without request data."""
        if direction not in {"to_blip3", "restore"}:
            direction = "restore"
        if outcome not in {"success", "failure"}:
            outcome = "failure"
        self.residency_transitions.labels(direction=direction, outcome=outcome).inc()
        self.residency_transition_duration.labels(direction=direction).observe(
            max(float(duration_seconds), 0.0)
        )

    def scrape(self) -> bytes:
        return generate_latest(self.registry)
