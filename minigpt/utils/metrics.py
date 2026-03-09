"""
Metrics Tracking
================

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import math
from collections import defaultdict
from typing import Dict, List


class MetricsTracker:
    """Track and aggregate training metrics."""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)

    def update(self, name: str, value: float):
        self._metrics[name].append(value)

    def average(self, name: str) -> float:
        values = self._metrics.get(name, [])
        return sum(values) / max(len(values), 1)

    def latest(self, name: str) -> float:
        values = self._metrics.get(name, [0.0])
        return values[-1]

    def reset(self, name: str = None):
        if name:
            self._metrics[name] = []
        else:
            self._metrics.clear()

    def summary(self) -> Dict[str, float]:
        return {k: self.average(k) for k in self._metrics}


def perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))  # Clamp to prevent overflow
