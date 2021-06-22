from neurodynex3.working_memory_network import wm_model
from neurodynex3.tools import plot_tools
import brian2 as b2
from neurodynex3.leaky_integrate_and_fire import LIF
from matplotlib import pyplot as plt


def question_external_poisson_population():
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, \
    spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = \
        wm_model.simulate_wm(
            sim_time=800. * b2.ms, poisson_firing_rate=2.2 * b2.Hz, sigma_weight_profile=20., Jpos_excit2excit=1.6)

    plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, t_min=0. * b2.ms)
    plt.show()


def question_weight_profile():
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, \
    spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, weight_profile_45 = \
        wm_model.simulate_wm(
            sim_time=800. * b2.ms, poisson_firing_rate=2.3 * b2.Hz, sigma_weight_profile=5., Jpos_excit2excit=6)
    plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, t_min=0. * b2.ms)
    plt.show()
    plt.figure()
    plt.plot(weight_profile_45)
    plt.show()


def question_integration_of_input():
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = \
        wm_model.simulate_wm(
            stimulus_center_deg=120, stimulus_width_deg=60, stimulus_strength=0.5 * b2.namp,
            t_stimulus_start=100 * b2.ms,
            t_stimulus_duration=200 * b2.ms, sim_time=500. * b2.ms)
    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit,
                                                                           voltage_monitor_excit, t_min=0. * b2.ms)

    plt.show()
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = \
        wm_model.simulate_wm(
            stimulus_center_deg=120, stimulus_width_deg=30, stimulus_strength=0.5 * b2.namp,
            t_stimulus_start=100 * b2.ms,
            t_stimulus_duration=200 * b2.ms, sim_time=500. * b2.ms)
    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit,
                                                                           voltage_monitor_excit, t_min=0. * b2.ms)
    plt.show()


def question_role_of_inhib_population():
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = \
        wm_model.simulate_wm(N_excitatory=1024, N_inhibitory=1, sigma_weight_profile=20,
                             stimulus_center_deg=120, stimulus_width_deg=30, stimulus_strength=0.5 * b2.namp,
                             t_stimulus_start=100 * b2.ms,
                             t_stimulus_duration=100 * b2.ms, sim_time=500. * b2.ms, Jpos_excit2excit=1.6,
                             poisson_firing_rate=1.5 * b2.Hz)
    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit,
                                                                           voltage_monitor_excit, t_min=0. * b2.ms)
    plt.show()


def distractor_test():
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = \
        wm_model.simulate_wm(N_excitatory=1024, N_inhibitory=256, sigma_weight_profile=40,
                             stimulus_center_deg=120, stimulus_width_deg=30, stimulus_strength=0.5 * b2.namp,
                             t_stimulus_start=100 * b2.ms,
                             t_stimulus_duration=100 * b2.ms, sim_time=500. * b2.ms, Jpos_excit2excit=1.6,
                             poisson_firing_rate=1.5 * b2.Hz, distractor_center_deg=270, distractor_width_deg=30,
                             distractor_strength=0.5 * b2.namp,
                             t_distractor_start=200 * b2.ms, t_distractor_duration=30 * b2.ms)
    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit,
                                                                           voltage_monitor_excit, t_min=0. * b2.ms)
    plt.show()


def distractor_at_same_time():
    rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = \
        wm_model.simulate_wm(N_excitatory=1024, N_inhibitory=256, sigma_weight_profile=60,
                             stimulus_center_deg=40, stimulus_width_deg=20, stimulus_strength=0.5 * b2.namp,
                             t_stimulus_start=100 * b2.ms,G_excit2inhib=0.355*b2.nS,
                             t_stimulus_duration=100 * b2.ms, sim_time=500. * b2.ms, Jpos_excit2excit=1.6,
                             poisson_firing_rate=1.5 * b2.Hz, distractor_center_deg=300, distractor_width_deg=20,
                             distractor_strength=0.5 * b2.namp,
                             t_distractor_start=110 * b2.ms, t_distractor_duration=100 * b2.ms)
    fig, ax_raster, ax_rate, ax_voltage = plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit,
                                                                           voltage_monitor_excit, t_min=0. * b2.ms)
    plt.show()


def mapping_neuron_preferred_direction():
    def get_orientation(idx_list, N):
        """
        Maps a vactor of neuron indices idx_list onto a vector of preferred direcitons
        :param idx_list: Subset of k monitored neurons
        :param N: Total number of neurons in the excitatory population
        :return: List of anglues that each idx represents in an N excitatory population
        """

        return [360 / N * i for i in idx_list]

    a = get_orientation([0, 1, 5, 10], 11)
    b = get_orientation([0, 1, 499, 500, 999], 1000)
    # SOMETHING IS WRONG WITH THE SOLUTION
    print(a, b)


def extracting_spikes():
    pass


if __name__ == '__main__':
    # rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = wm_model.simulate_wm(
    #     sim_time=800. * b2.ms, poisson_firing_rate=2.2 * b2.Hz, sigma_weight_profile=20., Jpos_excit2excit=1.6)
    # plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, t_min=0. * b2.ms)
    # plt.show()
    # wm_model.getting_started()
    # question_weight_profile()
    # question_integration_of_input()
    # question_role_of_inhib_population()
    # mapping_neuron_preferred_direction()
    # distractor_test()
    distractor_at_same_time()
