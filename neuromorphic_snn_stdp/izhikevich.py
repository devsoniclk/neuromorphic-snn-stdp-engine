"""Izhikevich 2D dynamical spiking neuron model."""

from typing import ClassVar

from .models import IzhikevichConfig


class IzhikevichNeuron:
    """Izhikevich 2D spiking neuron reproducing known cortical firing patterns."""

    PATTERNS: ClassVar[dict[str, IzhikevichConfig]] = {
        "regular_spiking": IzhikevichConfig(a=0.02, b=0.2, c=-65.0, d=8.0),
        "intrinsically_bursting": IzhikevichConfig(a=0.02, b=0.2, c=-55.0, d=4.0),
        "chattering": IzhikevichConfig(a=0.02, b=0.2, c=-50.0, d=2.0),
        "fast_spiking": IzhikevichConfig(a=0.1, b=0.2, c=-65.0, d=2.0),
        "low_threshold_spiking": IzhikevichConfig(a=0.02, b=0.25, c=-65.0, d=2.0),
        "resonator": IzhikevichConfig(a=0.1, b=0.26, c=-65.0, d=2.0),
    }

    def __init__(self, config: IzhikevichConfig | None = None, pattern_name: str | None = None):
        if pattern_name and pattern_name in self.PATTERNS:
            self.config = self.PATTERNS[pattern_name]
        else:
            self.config = config or IzhikevichConfig()
        self.v = self.config.c
        self.u = self.config.b * self.v
        self.total_spikes = 0
        self.last_spike_time: float | None = None

    def reset(self):
        """Reset dynamical state variables."""
        self.v = self.config.c
        self.u = self.config.b * self.v
        self.total_spikes = 0
        self.last_spike_time = None

    def step(self, dt_ms: float, i_inj: float = 0.0, current_time_ms: float = 0.0) -> bool:
        """
        Advance state using two-stage numerical integration for numerical stability.
        Returns True if an action potential spike is fired.
        """
        half_dt = 0.5 * dt_ms
        self.v += half_dt * (0.04 * (self.v**2) + 5.0 * self.v + 140.0 - self.u + i_inj)
        self.v += half_dt * (0.04 * (self.v**2) + 5.0 * self.v + 140.0 - self.u + i_inj)
        self.u += dt_ms * (self.config.a * (self.config.b * self.v - self.u))

        if self.v >= self.config.v_peak:
            self.v = self.config.c
            self.u += self.config.d
            self.last_spike_time = current_time_ms
            self.total_spikes += 1
            return True

        return False
