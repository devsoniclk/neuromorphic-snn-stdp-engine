"""Spike encoding generators for rate, temporal, and burst coding."""

import random


class SpikeEncoder:
    """Encodes continuous signals into neuromorphic discrete spike events."""

    @staticmethod
    def generate_poisson_train(
        rate_hz: float, duration_ms: float, dt_ms: float = 1.0, seed: int | None = None
    ) -> list[float]:
        """Generates Poisson spike arrival timestamps within [0, duration_ms]."""
        if seed is not None:
            random.seed(seed)
        p_spike = (rate_hz / 1000.0) * dt_ms
        spikes = []
        t = 0.0
        while t < duration_ms:
            if random.random() < p_spike:
                spikes.append(round(t, 2))
                t += 1.0  # Minimal refractory spacing
            else:
                t += dt_ms
        return spikes

    @staticmethod
    def latency_encode(
        value: float, min_val: float, max_val: float, max_latency_ms: float = 50.0
    ) -> float:
        """Time-to-first-spike temporal encoding. Higher value fires sooner."""
        norm = max(0.0, min(1.0, (value - min_val) / max(1e-6, max_val - min_val)))
        return round((1.0 - norm) * max_latency_ms, 2)

    @staticmethod
    def generate_burst_train(
        burst_count: int, intra_burst_isi_ms: float = 3.0, start_time_ms: float = 10.0
    ) -> list[float]:
        """Generates high-frequency burst trains for testing non-linear triplet STDP."""
        spikes = []
        t = start_time_ms
        for _ in range(burst_count):
            spikes.append(round(t, 2))
            t += intra_burst_isi_ms
        return spikes
