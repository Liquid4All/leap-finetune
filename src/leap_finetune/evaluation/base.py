"""Benchmark base class for evaluation during training.

Subclass ``Benchmark`` and implement ``load_samples()`` + ``evaluate()``.
The callback handles distributed sharding, all-reduce, wandb logging, and timing.

Benchmarks return a ``BenchmarkResult`` with a dict of named metrics — each
metric is averaged across all samples and logged to wandb separately.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Aggregated evaluation result. The callback all-reduces each metric across ranks.

    ``metrics`` is a dict of accumulated scores (e.g. {"accuracy": 42.0, "f1": 38.5}).
    The callback divides each by ``count`` to get averages.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    count: int = 0


class Benchmark(ABC):
    """Base class for all benchmarks.

    Subclass and implement:
      - ``load_samples()`` — return evaluation data (called once, cached).
      - ``evaluate(model, samples, device)`` — score the model, return BenchmarkResult.
    """

    def __init__(self, name: str):
        self.name = name
        self._samples: list | None = None

    @abstractmethod
    def load_samples(self) -> list:
        """Return all evaluation samples. Called once, then cached."""
        ...

    @abstractmethod
    def evaluate(self, model, samples: list, device) -> BenchmarkResult:
        """Score the model on ``samples`` (already sharded to this rank)."""
        ...

    def get_samples(self) -> list:
        if self._samples is None:
            self._samples = self.load_samples()
        return self._samples
