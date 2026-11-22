# Neuromorphic SNN & STDP Engine

[![CI](https://github.com/devsoniclk/neuromorphic-snn-stdp-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/devsoniclk/neuromorphic-snn-stdp-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Production-grade Spiking Neural Network (SNN) simulation and online learning engine with Leaky Integrate-and-Fire (LIF), 2D Izhikevich dynamical models, and Spike-Timing-Dependent Plasticity (STDP) synaptic homeostasis.

## Key Capabilities

1. **Biophysical Spiking Neuron Models**:
   - **Leaky Integrate-and-Fire (LIF)**: Membrane resistance $R_m$, membrane time constant $\tau_m$, subthreshold leakage, and absolute refractory period.
   - **Izhikevich 2D Dynamical System**: Exact bifurcations reproducing Regular Spiking (RS), Intrinsically Bursting (IB), Chattering (CH), Fast Spiking (FS), Low-Threshold Spiking (LTS), and Resonator (RZ).
2. **Synaptic Plasticity (STDP)**:
   - **Pairwise STDP**: Exponential causal window ($A_+, A_-, \tau_+, \tau_-$) with floating-point underflow protection.
   - **Triplet STDP**: Multi-trace pre/post dynamics ($r_1, r_2, o_1, o_2$) capturing burst potentiation.
   - **Three-Factor Eligibility Traces**: Temporal bridging between local spikes and delayed neuromodulatory reward signals.
3. **Homeostatic Synaptic Scaling**:
   - **Multiplicative Homeostatic Scaling**: Continuous weight adjustment preserving target firing rate setpoints.
   - **BCM Metaplasticity**: Sliding modification threshold $\theta_M = \mathbb{E}[R_{\text{post}}]^2 / R_0$.
4. **Protocols & Interfaces**:
   - **JSON-RPC 2.0 MCP Server**: FastMCP stdio server providing standard tools.
   - **Rich CLI**: Interactive visualization for single neurons, networks, and plasticity curves.

## Installation

```bash
git clone https://github.com/devsoniclk/neuromorphic-snn-stdp-engine.git
cd neuromorphic-snn-stdp-engine
uv venv
uv pip install -e ".[dev]"
```

## Quick Start

```python
from neuromorphic_snn_stdp import SpikingNetwork, LIFNeuron, STDPEngine

# 1. Simulate LIF Neuron
lif = LIFNeuron()
for t in range(50):
    if lif.step(dt_ms=1.0, i_inj=4.0, current_time_ms=float(t)):
        print(f"Spike emitted at t={t} ms")

# 2. Pairwise STDP Weight Update
stdp = STDPEngine()
delta_w = stdp.compute_pair_delta(dt_ms=10.0)  # Pre precedes Post -> LTP
print(f"STDP Delta W: {delta_w:+.6f}")

# 3. Simulate Multi-Neuron Network
net = SpikingNetwork(num_neurons=4)
net.fully_connect(initial_weight=0.5)
result = net.simulate(duration_ms=100.0, input_currents={0: 4.0})
print(f"Total Spikes: {result.total_spikes}, Mean Rate: {result.mean_firing_rate_hz} Hz")
```

## Running the CLI

```bash
# Simulate LIF neuron
snn-engine lif --duration 100 --current 3.5

# Simulate Izhikevich fast-spiking pattern
snn-engine izhikevich --pattern fast_spiking --duration 100

# Calculate STDP weight delta
snn-engine stdp --dt 15.0

# Run full network demo
snn-engine demo
```

## Testing

```bash
uv run pytest
```

## Author

**Lakshitha Dananjaya** ([@devsoniclk](https://github.com/devsoniclk))  
Co-authored-by: OpenBot `<335604065+openbotgit[bot]@users.noreply.github.com>`

## License

MIT License.
