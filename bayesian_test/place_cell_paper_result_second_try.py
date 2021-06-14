import brian2 as bs
from matplotlib import pyplot as plt
import numpy as np
from bayesian_test.utils import visualise_connectivity

w_input_to_place = 26.0 * bs.mV
TAU_M = 17.0 * bs.ms
TAU_S = 5.0 * bs.ms
threshold = -55.0 * bs.mV
v_rest = -80.0 * bs.mV
v_reset = -80.0 * bs.mV
e = 2.71828182846
bs.seed(seed=19971124)

dt = 0.1

lif = """
 dv/dt = (v_rest - v)/tau_m : volt (unless refractory)
 tau_m : second
 """

synaptic_update = """
 v_post += w_input_to_place
 """

threshold_eq = "v > (1+1/e)*(w_input_to_place + v_rest) + w_input_to_place"
reset_eq = "v = v_reset"

SIM_TIME = 5


def generate_gaussian_spike_train(mean, std):
    """
    https://brian2.readthedocs.io/en/stable/user/input.html use timed arrays
    """
    coef = 1 / (std * (2 * np.pi) ** (0.5))
    train = []
    time = np.linspace(0, SIM_TIME, num=int(SIM_TIME / dt))
    train = np.zeros_like(time)
    for i, t in enumerate(time):
        print(t)
        exp = -0.5 * ((t - mean) / std) ** 2
        rate = coef * np.exp(exp)
        train[i] = rate

    train = train * 40
    plt.plot(time, train)
    plt.xlabel('Time (ms)')
    plt.ylabel('Rates')
    plt.title(f"SpikeTrain")
    plt.show()
    return train


def interp_based(a, N=10):
    s = N
    l = (a.size - 1) * s + 1  # total length after interpolation
    return np.interp(np.arange(l), np.arange(l, step=s), a)


def simulation1():
    bs.start_scope()

    rates = [5, 10, 15]
    bs.store()

    train1 = generate_gaussian_spike_train(mean=1.5, std=1.7)
    train2 = generate_gaussian_spike_train(mean=3, std=1.7)
    # exit(0)
    train_len = train1.shape[0]
    # train_len = 10

    total_time = None
    total_spikes = None

    for i in range(train_len):
        r1 = train1[i]
        r2 = train2[i]
        place_cell = bs.NeuronGroup(1, model=lif, reset=reset_eq, threshold=threshold_eq, refractory=TAU_M,
                                    method="euler")

        place_cell.tau_m = TAU_M
        # place_cell.tau_s = TAU_S

        place_cell.v = -80.0 * bs.mV

        print(f"Rates: {r1, r2}")
        # bs.restore()
        input = bs.PoissonGroup(2, rates=np.array([r1, r2]) * bs.Hz)

        # connect input poisson spike generator to the input cells (grid and boundary vector)
        S1 = bs.Synapses(input, place_cell, on_pre=synaptic_update)
        S1.connect = S1.connect(i=[0, 1], j=0)
        step_per_time = 100
        place_cell_v_monitor = bs.StateMonitor(place_cell, 'v', record=True, dt=(dt / step_per_time) * bs.second)

        place_cell_monitor = bs.SpikeMonitor(source=place_cell)

        bs.run(dt * bs.second)

        spikes_i = place_cell_monitor.i
        spikes_t = place_cell_monitor.t

        print(spikes_i)
        print(spikes_t)

        if total_spikes is None:
            total_spikes = spikes_t / bs.ms
        else:
            total_spikes = np.concatenate([total_spikes, (i * step_per_time) + spikes_t / bs.ms])

        print("time", place_cell_v_monitor.t / bs.ms)
        if total_time is None:
            total_time = place_cell_v_monitor.t / bs.ms
        else:
            total_time = np.concatenate([total_time, (i * step_per_time) + place_cell_v_monitor.t / bs.ms])
    total_time = interp_based(total_time, N=10)
    print(type(total_time))
    print(total_time.shape)
    print(total_time)
    print(total_spikes)
    plt.figure()
    _, ind, _ = np.intersect1d(total_time, total_spikes, assume_unique=True, return_indices=True)
    spikes = np.zeros_like(total_time)
    spikes[ind] = 1
    plt.plot(total_time, spikes)
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.title(f"Spikes")
    plt.show()
    # print(spikes_i, spikes_t)
    # print(place_cell_v_monitor.v)
    # print(type(place_cell_v_monitor.t), type(place_cell_v_monitor.v[0]))
    # plt.figure()
    # plt.plot(place_cell_v_monitor.t / bs.ms, place_cell_v_monitor.v[0])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('v')
    # plt.title(f"Rates: {r1, r2}")
    # plt.show()


if __name__ == '__main__':
    simulation1()
