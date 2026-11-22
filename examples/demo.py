"""End-to-end demo of Neuromorphic SNN and STDP learning."""

from neuromorphic_snn_stdp import LIFNeuron, SpikingNetwork, STDPEngine


def main():
    print("=== 1. Simulating Leaky Integrate-and-Fire Neuron ===")
    lif = LIFNeuron()
    spikes = [t for t in range(100) if lif.step(1.0, 3.0, float(t))]
    print(f"LIF spikes in 100ms: {len(spikes)} spikes")

    print("\n=== 2. Pairwise STDP Plasticity Curve ===")
    stdp = STDPEngine()
    for dt in [-20.0, -10.0, -5.0, 5.0, 10.0, 20.0]:
        dw = stdp.compute_pair_delta(dt)
        print(f"dt = {dt:+5.1f} ms -> delta_w = {dw:+.6f}")

    print("\n=== 3. Multi-Neuron Network Simulation with STDP ===")
    net = SpikingNetwork(num_neurons=4)
    net.fully_connect(initial_weight=0.5)
    res = net.simulate(duration_ms=100.0, input_currents={0: 5.0})
    print(f"Total network spikes: {res.total_spikes}")
    print(f"Mean firing rate: {res.mean_firing_rate_hz} Hz")
    print(f"Sample weights: {res.final_weights[:4]}")


if __name__ == "__main__":
    main()
