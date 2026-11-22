from neuromorphic_snn_stdp.network import SpikingNetwork


def test_network_creation():
    net = SpikingNetwork(num_neurons=3)
    assert len(net.neurons) == 3
    net.connect(0, 1, 0.5)
    assert len(net.synapses) == 1


def test_network_fully_connect():
    net = SpikingNetwork(num_neurons=4)
    net.fully_connect(initial_weight=0.3)
    assert len(net.synapses) == 12


def test_network_simulation():
    net = SpikingNetwork(num_neurons=4)
    net.fully_connect(initial_weight=0.5)
    res = net.simulate(duration_ms=60.0, input_currents={0: 5.0})
    assert res.duration_ms == 60.0
    assert res.total_spikes >= 0
    assert len(res.final_weights) == 12
    for w in res.final_weights:
        assert 0.0 <= w <= 1.0
