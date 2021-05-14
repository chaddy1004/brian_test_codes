import brian2 as bs
from neurodynex3.leaky_integrate_and_fire import LIF


class GridCellLIF:
    def __init__(self, v_rest=-70,
                 v_reset=-65,
                 firing_threshold=-50,
                 membrane_resistance=10,
                 membrane_time_scale=8,
                 abs_refractory_period=2):
        V_REST = v_rest * bs.mV
        V_RESET = v_reset * bs.mV
        FIRING_THRESHOLD = firing_threshold * bs.mV
        MEMBRANE_RESISTANCE = membrane_resistance * bs.Mohm
        MEMBRANE_TIME_SCALE = membrane_time_scale * bs.ms
        ABSOLUTE_REFRACTORY_PERIOD = 2.0 * bs.ms
        eqs = """
        dv/dt =
        ( -(v-V_REST) + MEMBRANE_RESISTANCE * input_current(t,i) ) / MEMBRANE_TIME_SCALE : volt (unless refractory)
        """

        reset_eq = "v=V_RESET"

        threshold_eq = "v > FIRING_THRESHOLD"

        self.neuron = bs.NeuronGroup(1, model=eqs, reset=reset_eq, threshold=threshold_eq,
                                     refractory=ABSOLUTE_REFRACTORY_PERIOD, method="linear")

        self.neuron.v = v_rest




class BoundaryCellLIF:
    def __init__(self, v_rest=-70,
                 v_reset=-65,
                 firing_threshold=-50,
                 membrane_resistance=10,
                 membrane_time_scale=8,
                 abs_refractory_period=2):
        V_REST = v_rest * bs.mV
        V_RESET = v_reset * bs.mV
        FIRING_THRESHOLD = firing_threshold * bs.mV
        MEMBRANE_RESISTANCE = membrane_resistance * bs.Mohm
        MEMBRANE_TIME_SCALE = membrane_time_scale * bs.ms
        ABSOLUTE_REFRACTORY_PERIOD = 2.0 * bs.ms
        eqs = """
        dv/dt =
        ( -(v-V_REST) + MEMBRANE_RESISTANCE * input_current(t,i) ) / MEMBRANE_TIME_SCALE : volt (unless refractory)
        """

        reset_eq = "v=V_RESET"

        threshold_eq = "v > FIRING_THRESHOLD"

        self.neuron = bs.NeuronGroup(1, model=eqs, reset=reset_eq, threshold=threshold_eq,
                                     refractory=ABSOLUTE_REFRACTORY_PERIOD, method="linear")

        self.neuron.v = v_rest