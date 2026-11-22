"""Performance benchmark for neuromorphic SNN simulation."""

import time

from neuromorphic_snn_stdp import LIFNeuron, SpikingNetwork


def benchmark_lif():
    neuron = LIFNeuron()
    start = time.perf_counter()
    steps = 100000
    for i in range(steps):
        neuron.step(0.1, 3.0, i * 0.1)
    duration = time.perf_counter() - start
    print(f"LIF 100k steps: {duration:.4f}s ({steps / duration / 1000:.2f} ksteps/sec)")


def benchmark_network():
    net = SpikingNetwork(num_neurons=10)
    net.fully_connect(initial_weight=0.5)
    start = time.perf_counter()
    res = net.simulate(duration_ms=500.0, dt_ms=0.5, input_currents={0: 4.0, 1: 3.0})
    duration = time.perf_counter() - start
    print(f"Network (10 neurons, 90 synapses, 500ms): {duration:.4f}s ({res.total_spikes} spikes)")


if __name__ == "__main__":
    benchmark_lif()
    benchmark_network()
