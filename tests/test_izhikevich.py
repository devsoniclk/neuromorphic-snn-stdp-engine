from neuromorphic_snn_stdp.izhikevich import IzhikevichNeuron


def test_izhikevich_patterns_available():
    for name in ["regular_spiking", "intrinsically_bursting", "chattering", "fast_spiking"]:
        neuron = IzhikevichNeuron(pattern_name=name)
        assert neuron.v == neuron.config.c


def test_izhikevich_spiking():
    neuron = IzhikevichNeuron(pattern_name="regular_spiking")
    spikes = 0
    for t in range(100):
        if neuron.step(dt_ms=0.5, i_inj=10.0, current_time_ms=t * 0.5):
            spikes += 1
    assert spikes > 0
    assert neuron.total_spikes == spikes


def test_izhikevich_reset():
    neuron = IzhikevichNeuron(pattern_name="fast_spiking")
    neuron.step(dt_ms=0.5, i_inj=10.0)
    neuron.reset()
    assert neuron.total_spikes == 0
    assert neuron.v == neuron.config.c
