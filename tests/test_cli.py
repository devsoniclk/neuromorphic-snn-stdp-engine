from click.testing import CliRunner

from neuromorphic_snn_stdp.cli import cli


def test_cli_lif():
    runner = CliRunner()
    res = runner.invoke(cli, ["lif", "--duration", "50", "--current", "3.0"])
    assert res.exit_code == 0
    assert "LIF Neuron Simulation" in res.output


def test_cli_izhikevich():
    runner = CliRunner()
    res = runner.invoke(cli, ["izhikevich", "--pattern", "fast_spiking", "--duration", "50"])
    assert res.exit_code == 0
    assert "fast_spiking" in res.output


def test_cli_stdp():
    runner = CliRunner()
    res = runner.invoke(cli, ["stdp", "--dt", "10.0"])
    assert res.exit_code == 0
    assert "Delta w" in res.output


def test_cli_demo():
    runner = CliRunner()
    res = runner.invoke(cli, ["demo"])
    assert res.exit_code == 0
    assert "Spiking Network STDP Demo" in res.output
