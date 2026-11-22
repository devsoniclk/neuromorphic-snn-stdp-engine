"""Neuromorphic SNN and STDP simulation engine."""

from .encoders import SpikeEncoder
from .izhikevich import IzhikevichNeuron
from .lif import LIFNeuron
from .models import (
    HomeostaticConfig,
    IzhikevichConfig,
    LIFConfig,
    SimulationResult,
    SpikeEvent,
    STDPConfig,
)
from .network import SpikingNetwork
from .stdp import EligibilityTrace, HomeostaticManager, STDPEngine

__version__ = "0.1.0"
__all__ = [
    "EligibilityTrace",
    "HomeostaticConfig",
    "HomeostaticManager",
    "IzhikevichConfig",
    "IzhikevichNeuron",
    "LIFConfig",
    "LIFNeuron",
    "STDPConfig",
    "STDPEngine",
    "SimulationResult",
    "SpikeEncoder",
    "SpikeEvent",
    "SpikingNetwork",
]
