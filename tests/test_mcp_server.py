from neuromorphic_snn_stdp.mcp_server import FastMCPServer


def test_mcp_handlers():
    server = FastMCPServer()
    res_lif = server.handle_simulate_lif({"duration_ms": 50.0, "i_inj": 4.0})
    assert "total_spikes" in res_lif

    res_izh = server.handle_simulate_izhikevich({"pattern": "regular_spiking", "duration_ms": 50.0})
    assert res_izh["pattern"] == "regular_spiking"

    res_stdp = server.handle_pair_stdp({"dt_ms": 15.0})
    assert res_stdp["plasticity_type"] == "LTP"

    res_net = server.handle_simulate_network({"neurons": 3, "duration_ms": 30.0})
    assert "mean_firing_rate_hz" in res_net

    res_enc = server.handle_encode_poisson({"rate_hz": 20.0, "duration_ms": 50.0})
    assert "spikes" in res_enc
