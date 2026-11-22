"""Command Line Interface for Neuromorphic SNN STDP engine."""

import click
from rich.console import Console
from rich.table import Table

from .izhikevich import IzhikevichNeuron
from .lif import LIFNeuron
from .mcp_server import FastMCPServer
from .network import SpikingNetwork
from .stdp import STDPEngine

console = Console()


@click.group()
def cli():
    """Neuromorphic SNN STDP Engine CLI."""


@cli.command()
@click.option("--duration", default=100.0, help="Simulation duration in ms")
@click.option("--current", default=3.0, help="Injected current in nA")
def lif(duration: float, current: float):
    """Simulate single LIF neuron."""
    neuron = LIFNeuron()
    dt = 0.5
    spikes = []
    t = 0.0
    while t < duration:
        t += dt
        if neuron.step(dt, current, t):
            spikes.append(round(t, 2))

    table = Table(title="LIF Neuron Simulation")
    table.add_column("Duration (ms)", justify="center")
    table.add_column("Injected Current (nA)", justify="center")
    table.add_column("Total Spikes", justify="center")
    table.add_column("Firing Rate (Hz)", justify="center")
    table.add_row(
        str(duration),
        str(current),
        str(len(spikes)),
        str(round((len(spikes) / duration) * 1000.0, 2)),
    )
    console.print(table)


@cli.command()
@click.option("--pattern", default="regular_spiking", help="Izhikevich pattern name")
@click.option("--duration", default=100.0, help="Duration in ms")
@click.option("--current", default=10.0, help="Injected current")
def izhikevich(pattern: str, duration: float, current: float):
    """Simulate Izhikevich neuron pattern."""
    neuron = IzhikevichNeuron(pattern_name=pattern)
    dt = 0.5
    spikes = []
    t = 0.0
    while t < duration:
        t += dt
        if neuron.step(dt, current, t):
            spikes.append(round(t, 2))

    table = Table(title=f"Izhikevich Pattern: {pattern}")
    table.add_column("Pattern", justify="left")
    table.add_column("Duration (ms)", justify="center")
    table.add_column("Total Spikes", justify="center")
    table.add_column("Rate (Hz)", justify="center")
    table.add_row(
        pattern, str(duration), str(len(spikes)), str(round((len(spikes) / duration) * 1000.0, 2))
    )
    console.print(table)


@cli.command()
@click.option("--dt", default=10.0, help="Delta t in ms (t_post - t_pre)")
def stdp(dt: float):
    """Calculate pair STDP weight delta."""
    engine = STDPEngine()
    dw = engine.compute_pair_delta(dt)
    kind = "LTP" if dw > 0 else ("LTD" if dw < 0 else "neutral")
    console.print(f"[bold green]STDP dt={dt} ms -> Delta w = {round(dw, 6)} ({kind})[/bold green]")


@cli.command()
def demo():
    """Run interactive multi-neuron SNN demo with STDP."""
    net = SpikingNetwork(num_neurons=4)
    net.fully_connect(initial_weight=0.4)
    res = net.simulate(duration_ms=100.0, input_currents={0: 4.0, 1: 2.0})

    table = Table(title="Spiking Network STDP Demo")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("Duration", f"{res.duration_ms} ms")
    table.add_row("Total Spikes", str(res.total_spikes))
    table.add_row("Mean Firing Rate", f"{res.mean_firing_rate_hz} Hz")
    table.add_row("Final Synaptic Weights", str(res.final_weights[:6]) + "...")
    console.print(table)


@cli.command()
def mcp():
    """Run FastMCP JSON-RPC stdio server."""
    server = FastMCPServer()
    server.run()


if __name__ == "__main__":
    cli()
