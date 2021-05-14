import brian2 as bs
from matplotlib import pyplot as plt
import numpy as np
from bayesian_test.base_simulation import BaseSimulation

w_input_to_place = 26.0 * bs.mV
TAU_M = 17.0 * bs.ms
TAU_S = 5.0 * bs.ms
threshold = -55.0 * bs.mV
v_rest = -80.0 * bs.mV
v_reset = -80.0 * bs.mV

# w_input_to_place = 1000.0
# TAU_M = 17.0 * bs.ms
# TAU_S = 5.0 * bs.ms
# threshold = -55.0
# v_rest = -80.0
# v_reset = -80.0

# lif = """
#  dv/dt = (v_rest - v + J)/TAU_M : volt
#  dJ/dt = -J/TAU_S: volt
#  tau_m : second
#  tau_s: second
#  """

lif = """
 dv/dt = (v_rest - v + J)/tau_m : volt
 dJ/dt = -J/tau_s: volt
 tau_m : second
 tau_s: second
 """

synaptic_update = """
 J_post += ((TAU_M - TAU_S)/TAU_M)*w_input_to_place
 """

threshold_eq = "v > threshold"
reset_eq = "v = v_reset"


def define_model():
    bs.start_scope()

    place_cell = bs.NeuronGroup(1, model=lif, reset=reset_eq, threshold=threshold_eq,
                                method="euler")

    place_cell.tau_m = TAU_M
    place_cell.tau_s = TAU_S

    place_cell.v = -80.0 * bs.mV

    # connect input poisson spike generator to the input cells (grid and boundary vector)
    input = bs.PoissonGroup(2, rates=np.array([10, 20]) * bs.Hz)
    # boundary_cell_input = bs.PoissonGroup(1, rates=20*bs.Hz)
    print(synaptic_update)
    S1 = bs.Synapses(input, place_cell, on_pre=synaptic_update)
    S1.connect = S1.connect(i=[0, 1], j=0)

    place_cell_v_monitor = bs.StateMonitor(place_cell, 'v', record=True)

    place_cell_monitor = bs.SpikeMonitor(source=place_cell)

    bs.run(5 * bs.second)

    spikes_i = place_cell_monitor.i
    spikes_t = place_cell_monitor.t

    print(spikes_i, spikes_t)
    print(place_cell_v_monitor.v)

    plt.plot(place_cell_v_monitor.t / bs.ms, place_cell_v_monitor.v[0])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()


if __name__ == '__main__':
    define_model()
    # simulation1 = Simulation1(config=None)
    # simulation1.define_model()
    # # simulation1.run_simulation(time=1)
