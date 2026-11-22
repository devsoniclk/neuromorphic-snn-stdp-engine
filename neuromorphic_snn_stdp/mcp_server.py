"""JSON-RPC 2.0 stdio MCP server for Neuromorphic SNN and STDP engine."""

import json
import sys
from typing import Any

from .encoders import SpikeEncoder
from .izhikevich import IzhikevichNeuron
from .lif import LIFNeuron
from .network import SpikingNetwork
from .stdp import STDPEngine


class FastMCPServer:
    """Stdio JSON-RPC 2.0 MCP server providing neuromorphic tools."""

    def __init__(self):
        self.tools = {
            "snn_simulate_lif": self.handle_simulate_lif,
            "snn_simulate_izhikevich": self.handle_simulate_izhikevich,
            "snn_pair_stdp": self.handle_pair_stdp,
            "snn_simulate_network": self.handle_simulate_network,
            "snn_encode_poisson": self.handle_encode_poisson,
        }

    def handle_simulate_lif(self, args: dict[str, Any]) -> dict[str, Any]:
        duration_ms = float(args.get("duration_ms", 100.0))
        dt_ms = float(args.get("dt_ms", 1.0))
        i_inj = float(args.get("i_inj", 3.0))
        neuron = LIFNeuron()
        spikes = []
        t = 0.0
        while t < duration_ms:
            t += dt_ms
            if neuron.step(dt_ms, i_inj, t):
                spikes.append(round(t, 2))
        return {
            "total_spikes": len(spikes),
            "firing_rate_hz": round((len(spikes) / duration_ms) * 1000.0, 2),
            "spike_timestamps": spikes,
        }

    def handle_simulate_izhikevich(self, args: dict[str, Any]) -> dict[str, Any]:
        pattern = str(args.get("pattern", "regular_spiking"))
        duration_ms = float(args.get("duration_ms", 100.0))
        dt_ms = float(args.get("dt_ms", 0.5))
        i_inj = float(args.get("i_inj", 10.0))
        neuron = IzhikevichNeuron(pattern_name=pattern)
        spikes = []
        t = 0.0
        while t < duration_ms:
            t += dt_ms
            if neuron.step(dt_ms, i_inj, t):
                spikes.append(round(t, 2))
        return {
            "pattern": pattern,
            "total_spikes": len(spikes),
            "firing_rate_hz": round((len(spikes) / duration_ms) * 1000.0, 2),
            "spike_timestamps": spikes,
        }

    def handle_pair_stdp(self, args: dict[str, Any]) -> dict[str, Any]:
        dt_ms = float(args.get("dt_ms", 10.0))
        engine = STDPEngine()
        delta_w = engine.compute_pair_delta(dt_ms)
        return {
            "dt_ms": dt_ms,
            "delta_w": round(delta_w, 6),
            "plasticity_type": "LTP" if delta_w > 0 else ("LTD" if delta_w < 0 else "neutral"),
        }

    def handle_simulate_network(self, args: dict[str, Any]) -> dict[str, Any]:
        neurons = int(args.get("neurons", 4))
        duration_ms = float(args.get("duration_ms", 50.0))
        i_inj = float(args.get("i_inj", 3.0))
        net = SpikingNetwork(num_neurons=neurons)
        net.fully_connect(initial_weight=0.5)
        res = net.simulate(duration_ms=duration_ms, input_currents={0: i_inj})
        return {
            "duration_ms": res.duration_ms,
            "total_spikes": res.total_spikes,
            "mean_firing_rate_hz": res.mean_firing_rate_hz,
            "final_weights": res.final_weights,
        }

    def handle_encode_poisson(self, args: dict[str, Any]) -> dict[str, Any]:
        rate_hz = float(args.get("rate_hz", 20.0))
        duration_ms = float(args.get("duration_ms", 100.0))
        spikes = SpikeEncoder.generate_poisson_train(rate_hz, duration_ms)
        return {"rate_hz": rate_hz, "count": len(spikes), "spikes": spikes}

    def run(self):
        """Stdio processing loop."""
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                method = req.get("method")
                req_id = req.get("id")
                if method in self.tools:
                    result = self.tools[method](req.get("params", {}))
                    sys.stdout.write(
                        json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n"
                    )
                    sys.stdout.flush()
                else:
                    sys.stdout.write(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32601, "message": "Method not found"},
                            }
                        )
                        + "\n"
                    )
                    sys.stdout.flush()
            except Exception as e:  # noqa: BLE001
                sys.stdout.write(
                    json.dumps(
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
                    )
                    + "\n"
                )
                sys.stdout.flush()
