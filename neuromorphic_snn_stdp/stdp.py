"""Spike-Timing-Dependent Plasticity (STDP) and homeostatic scaling engine."""

import math

from .models import HomeostaticConfig, STDPConfig


class STDPEngine:
    """Computes pair-based and triplet STDP synaptic weight updates."""

    def __init__(self, config: STDPConfig | None = None):
        self.config = config or STDPConfig()

    def compute_pair_delta(self, dt_ms: float) -> float:
        """
        Calculates pairwise STDP weight update:
        dt = t_post - t_pre
        dt > 0: Pre precedes Post -> LTP
        dt < 0: Post precedes Pre -> LTD
        """
        if dt_ms > 0.0:
            exp_val = max(-dt_ms / self.config.tau_plus, -50.0)
            return self.config.a_plus * math.exp(exp_val)
        elif dt_ms < 0.0:
            exp_val = max(dt_ms / self.config.tau_minus, -50.0)
            return -self.config.a_minus * math.exp(exp_val)
        return 0.0

    def compute_triplet_step(
        self,
        r1: float,
        r2: float,
        o1: float,
        o2: float,
        is_post_spike: bool,
    ) -> float:
        """
        Triplet STDP formulation:
        Post-synaptic spike: dw = r1 * (a2_plus + a3_plus * o2)
        Pre-synaptic spike:  dw = -o1 * (a2_minus + a3_minus * r2)
        """
        if is_post_spike:
            return r1 * (self.config.a2_plus + self.config.a3_plus * o2)
        else:
            return -o1 * (self.config.a2_minus + self.config.a3_minus * r2)

    def clamp(self, weight: float) -> float:
        """Hard non-negative biological clamping."""
        return max(self.config.w_min, min(self.config.w_max, weight))


class EligibilityTrace:
    """Maintains exponential eligibility traces for 3-factor learning."""

    def __init__(self, tau_eligibility_ms: float = 100.0):
        self.tau_e = tau_eligibility_ms
        self.trace = 0.0

    def decay(self, dt_ms: float):
        """Exponential decay over dt_ms."""
        if dt_ms > 0.0:
            decay_factor = math.exp(max(-dt_ms / self.tau_e, -50.0))
            self.trace *= decay_factor

    def add_event(self, delta: float):
        """Accumulate instantaneous plasticity event."""
        self.trace += delta

    def compute_reward_update(self, reward: float, learning_rate: float = 0.1) -> float:
        """Three-factor update: Delta_w = learning_rate * reward * trace."""
        return learning_rate * reward * self.trace


class HomeostaticManager:
    """Homeostatic synaptic scaling and BCM sliding threshold metaplasticity."""

    def __init__(self, config: HomeostaticConfig | None = None):
        self.config = config or HomeostaticConfig()

    def multiplicative_scale(self, weight: float, current_rate_hz: float) -> float:
        """dw_homeo = scaling_rate * (target_rate - current_rate) * weight."""
        return self.config.scaling_rate * (self.config.target_rate_hz - current_rate_hz) * weight

    def bcm_threshold(self, history_rates: list[float]) -> float:
        """theta_M = E[rate]^2 / baseline_rate."""
        if not history_rates:
            return self.config.bcm_baseline_rate
        mean_rate = sum(history_rates) / len(history_rates)
        return (mean_rate**2) / (self.config.bcm_baseline_rate + 1e-6)
