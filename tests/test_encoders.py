from neuromorphic_snn_stdp.encoders import SpikeEncoder


def test_poisson_generator():
    spikes = SpikeEncoder.generate_poisson_train(rate_hz=50.0, duration_ms=200.0, seed=42)
    assert isinstance(spikes, list)
    assert len(spikes) > 0


def test_latency_encode():
    lat_high = SpikeEncoder.latency_encode(
        value=9.0, min_val=0.0, max_val=10.0, max_latency_ms=50.0
    )
    lat_low = SpikeEncoder.latency_encode(value=1.0, min_val=0.0, max_val=10.0, max_latency_ms=50.0)
    assert lat_high < lat_low


def test_burst_generator():
    burst = SpikeEncoder.generate_burst_train(
        burst_count=5, intra_burst_isi_ms=2.0, start_time_ms=10.0
    )
    assert len(burst) == 5
    assert burst[0] == 10.0
    assert burst[-1] == 18.0
