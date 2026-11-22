"""Leaky Integrate-and-Fire (LIF) neuron implementation."""

from .models import LIFConfig


class LIFNeuron:
    """Leaky Integrate-and-Fire spiking neuron with absolute refractory dynamics."""

    def __init__(self, config: LIFConfig | None = None):
        self.config = config or LIFConfig()
        self.v = self.config.v_rest
        self.refractory_time_remaining = 0.0
        self.last_spike_time: float | None = None
        self.total_spikes = 0

    def reset(self):
        """Reset internal membrane state and spike counters."""
        self.v = self.config.v_rest
        self.refractory_time_remaining = 0.0
        self.last_spike_time = None
        self.total_spikes = 0

    def step(self, dt_ms: float, i_inj: float = 0.0, current_time_ms: float = 0.0) -> bool:
        """Advance neuron state by time step dt_ms. Returns True if spike occurred."""
        if self.refractory_time_remaining > 0.0:
            self.refractory_time_remaining = max(0.0, self.refractory_time_remaining - dt_ms)
            self.v = self.config.v_reset
            return False

        # Continuous subthreshold integration: dv/dt = (-(v - v_rest) + R * I) / tau_m
        dv = (-(self.v - self.config.v_rest) + self.config.r_membrane * i_inj) * (
            dt_ms / self.config.tau_m
        )
        self.v += dv

        if self.v >= self.config.v_thresh:
            self.v = self.config.v_reset
            self.refractory_time_remaining = self.config.t_refractory
            self.last_spike_time = current_time_ms
            self.total_spikes += 1
            return True

        return False
