"""Data models and type definitions for neuromorphic SNN and STDP engine."""

from pydantic import BaseModel, Field


class LIFConfig(BaseModel):
    """Configuration for Leaky Integrate-and-Fire neuron."""

    tau_m: float = Field(20.0, description="Membrane time constant in ms")
    v_rest: float = Field(-70.0, description="Resting membrane potential in mV")
    v_reset: float = Field(-75.0, description="Reset membrane potential in mV")
    v_thresh: float = Field(-55.0, description="Spike threshold in mV")
    r_membrane: float = Field(10.0, description="Membrane resistance in MOhm")
    t_refractory: float = Field(2.0, description="Absolute refractory period in ms")


class IzhikevichConfig(BaseModel):
    """Parameters for 2D Izhikevich dynamical neuron."""

    a: float = Field(0.02, description="Time scale of recovery variable u")
    b: float = Field(0.2, description="Sensitivity of u to subthreshold fluctuations of v")
    c: float = Field(-65.0, description="After-spike reset value of v in mV")
    d: float = Field(8.0, description="After-spike reset increment of u")
    v_peak: float = Field(30.0, description="Peak membrane cutoff potential in mV")


class STDPConfig(BaseModel):
    """Configuration for pair and triplet STDP rules."""

    tau_plus: float = Field(20.0, description="LTP time constant in ms")
    tau_minus: float = Field(20.0, description="LTD time constant in ms")
    a_plus: float = Field(0.01, description="LTP amplitude")
    a_minus: float = Field(0.0105, description="LTD amplitude")
    tau_x: float = Field(101.0, description="Slow pre-synaptic trace time constant in ms")
    tau_y: float = Field(125.0, description="Slow post-synaptic trace time constant in ms")
    a2_plus: float = Field(0.005, description="Pairwise LTP amplitude in triplet rule")
    a3_plus: float = Field(0.012, description="Triplet LTP amplitude")
    a2_minus: float = Field(0.007, description="Pairwise LTD amplitude in triplet rule")
    a3_minus: float = Field(0.0001, description="Triplet LTD amplitude")
    w_min: float = Field(0.0, description="Minimum allowed synaptic weight")
    w_max: float = Field(1.0, description="Maximum allowed synaptic weight")


class HomeostaticConfig(BaseModel):
    """Homeostatic synaptic scaling and metaplasticity parameters."""

    target_rate_hz: float = Field(15.0, description="Target post-synaptic firing rate in Hz")
    scaling_rate: float = Field(0.001, description="Homeostatic scaling adjustment rate")
    bcm_baseline_rate: float = Field(15.0, description="BCM baseline rate for sliding threshold")


class SpikeEvent(BaseModel):
    """Spike occurrence event."""

    neuron_id: int
    timestamp_ms: float


class SimulationResult(BaseModel):
    """Summary of SNN simulation run."""

    duration_ms: float
    total_spikes: int
    mean_firing_rate_hz: float
    spikes: list[SpikeEvent]
    final_weights: list[float] | None = None
