"""Multi-neuron Spiking Neural Network simulation engine."""

from .lif import LIFNeuron
from .models import HomeostaticConfig, LIFConfig, SimulationResult, SpikeEvent, STDPConfig
from .stdp import EligibilityTrace, HomeostaticManager, STDPEngine


class Synapse:
    """Synaptic junction connecting a pre-synaptic to a post-synaptic neuron."""

    def __init__(
        self,
        pre_id: int,
        post_id: int,
        initial_weight: float = 0.5,
        config: STDPConfig | None = None,
    ):
        self.pre_id = pre_id
        self.post_id = post_id
        self.weight = initial_weight
        self.stdp_engine = STDPEngine(config)
        self.eligibility = EligibilityTrace()
        self.r1 = 0.0
        self.r2 = 0.0


class SpikingNetwork:
    """Network of spiking neurons with dynamic STDP synaptic connections."""

    def __init__(
        self,
        num_neurons: int = 5,
        neuron_config: LIFConfig | None = None,
        stdp_config: STDPConfig | None = None,
        homeostatic_config: HomeostaticConfig | None = None,
    ):
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron(neuron_config) for _ in range(num_neurons)]
        self.stdp_config = stdp_config or STDPConfig()
        self.stdp_engine = STDPEngine(self.stdp_config)
        self.homeo_manager = HomeostaticManager(homeostatic_config)
        self.synapses: list[Synapse] = []
        self.o1 = [0.0] * num_neurons
        self.o2 = [0.0] * num_neurons

    def connect(self, pre_id: int, post_id: int, initial_weight: float = 0.5):
        """Creates a directional synaptic connection."""
        if 0 <= pre_id < self.num_neurons and 0 <= post_id < self.num_neurons:
            self.synapses.append(Synapse(pre_id, post_id, initial_weight, self.stdp_config))

    def fully_connect(self, initial_weight: float = 0.5, allow_self_connections: bool = False):
        """Creates all-to-all synaptic connectivity."""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j or allow_self_connections:
                    self.connect(i, j, initial_weight)

    def simulate(
        self,
        duration_ms: float = 100.0,
        dt_ms: float = 1.0,
        input_currents: dict[int, float] | None = None,
        enable_stdp: bool = True,
        enable_homeostasis: bool = True,
    ) -> SimulationResult:
        """Executes continuous time-stepped simulation."""
        input_currents = input_currents or {}
        spikes: list[SpikeEvent] = []
        current_time = 0.0
        total_steps = int(duration_ms / dt_ms)

        for _ in range(total_steps):
            current_time += dt_ms
            step_spikes: dict[int, bool] = {}

            # Decay post traces
            for n_id in range(self.num_neurons):
                decay_minus = -dt_ms / self.stdp_config.tau_minus
                self.o1[n_id] *= 2.718281828459045 ** max(decay_minus, -50.0)
                decay_y = -dt_ms / self.stdp_config.tau_y
                self.o2[n_id] *= 2.718281828459045 ** max(decay_y, -50.0)

            # Decay pre traces on synapses
            for syn in self.synapses:
                decay_plus = -dt_ms / self.stdp_config.tau_plus
                syn.r1 *= 2.718281828459045 ** max(decay_plus, -50.0)
                decay_x = -dt_ms / self.stdp_config.tau_x
                syn.r2 *= 2.718281828459045 ** max(decay_x, -50.0)
                syn.eligibility.decay(dt_ms)

            # Step neurons with external + synaptic input
            for n_id, neuron in enumerate(self.neurons):
                i_inj = input_currents.get(n_id, 0.0)
                fired = neuron.step(dt_ms, i_inj, current_time)
                step_spikes[n_id] = fired
                if fired:
                    spikes.append(SpikeEvent(neuron_id=n_id, timestamp_ms=round(current_time, 2)))
                    self.o1[n_id] += 1.0
                    self.o2[n_id] += 1.0

            # Synaptic transmission and STDP updates
            for syn in self.synapses:
                pre_fired = step_spikes[syn.pre_id]
                post_fired = step_spikes[syn.post_id]

                if pre_fired:
                    syn.r1 += 1.0
                    syn.r2 += 1.0
                    # Depolarize post-synaptic target
                    self.neurons[syn.post_id].v += syn.weight * 2.0
                    if enable_stdp:
                        # LTD depression: pre arrives while post recently active
                        dw = self.stdp_engine.compute_triplet_step(
                            syn.r1,
                            syn.r2,
                            self.o1[syn.post_id],
                            self.o2[syn.post_id],
                            is_post_spike=False,
                        )
                        syn.weight = self.stdp_engine.clamp(syn.weight + dw)
                        syn.eligibility.add_event(dw)

                if post_fired and enable_stdp:
                    # LTP potentiation: post fires while pre trace is active
                    dw = self.stdp_engine.compute_triplet_step(
                        syn.r1,
                        syn.r2,
                        self.o1[syn.post_id],
                        self.o2[syn.post_id],
                        is_post_spike=True,
                    )
                    syn.weight = self.stdp_engine.clamp(syn.weight + dw)
                    syn.eligibility.add_event(dw)

            # Periodic homeostatic scaling
            if enable_homeostasis and int(current_time) % 50 == 0:
                for syn in self.synapses:
                    post_neuron = self.neurons[syn.post_id]
                    est_rate = (post_neuron.total_spikes / max(1e-3, current_time)) * 1000.0
                    dh = self.homeo_manager.multiplicative_scale(syn.weight, est_rate)
                    syn.weight = self.stdp_engine.clamp(syn.weight + dh)

        total_spikes = len(spikes)
        mean_rate = (total_spikes / max(1e-3, duration_ms * self.num_neurons)) * 1000.0
        final_weights = [round(syn.weight, 4) for syn in self.synapses]

        return SimulationResult(
            duration_ms=duration_ms,
            total_spikes=total_spikes,
            mean_firing_rate_hz=round(mean_rate, 2),
            spikes=spikes,
            final_weights=final_weights,
        )
