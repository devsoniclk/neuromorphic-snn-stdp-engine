import io
import json
from unittest.mock import patch

from neuromorphic_snn_stdp.izhikevich import IzhikevichNeuron
from neuromorphic_snn_stdp.mcp_server import FastMCPServer
from neuromorphic_snn_stdp.models import HomeostaticConfig, IzhikevichConfig
from neuromorphic_snn_stdp.network import SpikingNetwork
from neuromorphic_snn_stdp.stdp import HomeostaticManager


def test_izhikevich_custom_config():
    cfg = IzhikevichConfig(a=0.03, b=0.22, c=-60.0, d=6.0)
    neuron = IzhikevichNeuron(config=cfg)
    assert neuron.config.a == 0.03
    assert neuron.v == -60.0


def test_homeostatic_bcm_empty():
    manager = HomeostaticManager(HomeostaticConfig(bcm_baseline_rate=20.0))
    thresh = manager.bcm_threshold([])
    assert thresh == 20.0


def test_mcp_server_run_loop():
    server = FastMCPServer()
    input_data = (
        json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "snn_pair_stdp", "params": {"dt_ms": 10.0}}
        )
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "unknown_tool", "params": {}})
        + "\n"
        + "invalid_json\n"
        + "\n"
    )
    with (
        patch("sys.stdin", io.StringIO(input_data)),
        patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
    ):
        server.run()
        output = mock_stdout.getvalue()
        assert "jsonrpc" in output
        lines = output.strip().split("\n")
        assert len(lines) == 3


def test_network_self_connections():
    net = SpikingNetwork(num_neurons=2)
    net.fully_connect(initial_weight=0.6, allow_self_connections=True)
    assert len(net.synapses) == 4
    res = net.simulate(duration_ms=40.0, dt_ms=0.5, input_currents={0: 5.0, 1: 5.0})
    assert res.duration_ms == 40.0
