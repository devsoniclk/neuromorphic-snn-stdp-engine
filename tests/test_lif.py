from neuromorphic_snn_stdp.lif import LIFNeuron


def test_lif_initial_state():
    neuron = LIFNeuron()
    assert neuron.v == -70.0
    assert neuron.refractory_time_remaining == 0.0
    assert neuron.total_spikes == 0


def test_lif_subthreshold_integration():
    neuron = LIFNeuron()
    neuron.step(dt_ms=1.0, i_inj=1.0)
    assert neuron.v > -70.0
    assert neuron.v < -55.0


def test_lif_spike_and_reset():
    neuron = LIFNeuron()
    spiked = False
    for t in range(50):
        if neuron.step(dt_ms=1.0, i_inj=5.0, current_time_ms=float(t)):
            spiked = True
            break
    assert spiked is True
    assert neuron.v == -75.0
    assert neuron.refractory_time_remaining > 0.0
    assert neuron.total_spikes == 1


def test_lif_refractory_period():
    neuron = LIFNeuron()
    for t in range(30):
        if neuron.step(dt_ms=1.0, i_inj=10.0, current_time_ms=float(t)):
            break
    v_during_ref = neuron.v
    neuron.step(dt_ms=0.5, i_inj=10.0)
    assert neuron.v == v_during_ref


def test_lif_reset():
    neuron = LIFNeuron()
    neuron.step(dt_ms=1.0, i_inj=10.0)
    neuron.reset()
    assert neuron.v == -70.0
    assert neuron.total_spikes == 0
