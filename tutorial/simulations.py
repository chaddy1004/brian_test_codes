import brian2 as bs
from matplotlib import pyplot as plt
import time


def visualise_connectivity(S):
    plt.clf()
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(bs.zeros(Ns), bs.arange(Ns), 'ok', ms=10)
    plt.plot(bs.ones(Nt), bs.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
    plt.show()


def main1(plot=True):
    # running multiple separate simulations
    # this simulation checks to see the effect that different time constant have in LIF neuron firing given poisson spikes
    bs.start_scope()
    num_inputs = 100
    input_rate = 10 * bs.Hz
    weight = 0.1

    # range of time constants
    tau_range = bs.linspace(start=1, stop=10, num=30) * bs.ms
    # storing output rates
    output_rates = []

    for tau in tau_range:
        # introducing some external spike source
        P = bs.PoissonGroup(num_inputs, rates=input_rate)
        eqs = """
         dv/dt = -v/tau : 1
         """
        G = bs.NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
        # connect the external spike source with the neurons
        # instead of connecting the same neurons like in previous examples
        S = bs.Synapses(P, G, on_pre='v += weight')
        S.connect()
        spike_monitor = bs.SpikeMonitor(source=G)
        bs.run(1 * bs.second)
        n_spikes = spike_monitor.num_spikes
        output_rates.append(n_spikes / bs.second)

    if plot:
        plt.plot(tau_range / bs.ms, output_rates)
        plt.xlabel(r'$\tau$ (ms)')
        plt.ylabel('Firing Rate (spikes/s)')
        plt.show()


def main2(plot=True):
    # Optimizing the code from main1 by only making the networks just once before the loop
    # running multiple separate simulations
    bs.start_scope()
    num_inputs = 100
    input_rate = 10 * bs.Hz
    weight = 0.1

    # range of time constants
    tau_range = bs.linspace(start=1, stop=10, num=30) * bs.ms
    # storing output rates
    output_rates = []

    P = bs.PoissonGroup(num_inputs, rates=input_rate)
    eqs = """
     dv/dt = -v/tau : 1
     """
    G = bs.NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
    # connect the external spike source with the neurons
    # instead of connecting the same neurons like in previous examples
    S = bs.Synapses(P, G, on_pre='v += weight')
    S.connect()

    # visualise_connectivity(S)
    spike_monitor = bs.SpikeMonitor(source=G)
    bs.store()  # stores the current state of the network
    for tau in tau_range:
        # calling this allows for resetting of the network to its original state
        bs.restore()
        bs.run(1 * bs.second)
        n_spikes = spike_monitor.num_spikes
        output_rates.append(n_spikes / bs.second)
    if plot:
        plt.clf()
        plt.plot(tau_range / bs.ms, output_rates)
        plt.xlabel(r'$\tau$ (ms)')
        plt.ylabel('Firing Rate (spikes/s)')
        plt.show()


def main3(plot=True):
    # even faster
    # will not always work
    # Since there is only single output neuron in the model above,
    # we can make multiple output neurons and make each time constants a parameter of the grop
    bs.start_scope()
    num_inputs = 100
    input_rate = 10 * bs.Hz
    w = 0.1
    tau_range = bs.linspace(1, 10, 30) * bs.ms
    num_tau = len(tau_range)

    P = bs.PoissonGroup(num_inputs, rates=input_rate)

    eqs = """
    dv/dt = -v/tau : 1
    tau : second
    """

    # set output neurons with the same number of taus that you want to try with
    G = bs.NeuronGroup(num_tau, eqs, threshold='v>1', reset='v=0', method='exact')
    # set the taus for each individual neurons separately
    # (didn't know that this was possible)
    G.tau = tau_range
    S = bs.Synapses(P, G, on_pre='v += w')
    S.connect()
    spike_monitor = bs.SpikeMonitor(G)
    # Now we can just run once with no loop
    bs.run(1 * bs.second)
    # and each counts correspond to neuron with different taus
    output_rates = spike_monitor.count / bs.second  # firing rate is count/duration
    if plot:
        plt.plot(tau_range / bs.ms, output_rates)
        plt.xlabel(r'$\tau$ (ms)')
        plt.ylabel('Firing rate (sp/s)')
        plt.show()
        # this is significantly faster

def main4(plot=True):
    # what if we want to keep the input spike exactly the same throughout different taus?
    # Solution: We run PoissonGroup once, store all the spikes, and use the stored spikes across the multiple runs
    bs.start_scope()
    num_inputs = 100
    input_rate = 10 * bs.Hz
    w = 0.1
    tau_range = bs.linspace(1, 10, 30) * bs.ms
    output_rates = []

    P = bs.PoissonGroup(num_inputs, rates=input_rate)
    p_monitor = bs.SpikeMonitor(P)

    one_second = 1 * bs.second

    """
    Note that in the code above, we created Network objects. 
    The reason is that in the loop, if we just called run it would try to simulate all the objects, 
    including the Poisson neurons P, and we only want to run that once. 
    We use Network to specify explicitly which objects we want to include.
    """
    net = bs.Network(P, p_monitor)
    net.run(one_second)

    # keeps a copy of the spikes that are generated by the PoissonGroup during that explicit run earlier
    spikes_i = p_monitor.i
    spikes_t = p_monitor.t

    # Construct network that we run each time
    sgg = bs.SpikeGeneratorGroup(num_inputs, spikes_i, spikes_t)
    eqs = '''
    dv/dt = -v/tau : 1
    '''
    G = bs.NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
    S = bs.Synapses(sgg, G, on_pre='v += w')
    S.connect()  # fully connected network
    g_monitor = bs.SpikeMonitor(G)

    # store the current state of the network
    net = bs.Network(sgg, G, S, g_monitor)
    net.store()

    for tau in tau_range:
        net.restore()
        net.run(one_second)
        output_rates.append(g_monitor.num_spikes / bs.second)
    if plot:
        plt.clf()
        plt.plot(tau_range / bs.ms, output_rates)
        plt.xlabel(r'$\tau$ (ms)')
        plt.ylabel('Firing Rate (spikes/s)')
        # there is much less noise compared to before where we used different PoissonGroup everytime
        plt.show()


def compare_times():
    curr_time = time.time()
    main1(plot=False)
    end_time = time.time()
    t_1 = end_time - curr_time

    curr_time = time.time()
    main2(plot=False)
    end_time = time.time()
    t_2 = end_time - curr_time

    curr_time = time.time()
    main3(plot=False)
    end_time = time.time()
    t_3 = end_time - curr_time

    print(f"1-Naive: {t_1}")
    print(f"Making Groups only once (on average the best way): {t_2}")
    print(f"Trick (does not always work): {t_3}")

    """
    1-Naive: 13.791414022445679
    Making Groups only once (on average the best way): 12.281008005142212
    Trick (does not always work): 7.310742139816284
    """


if __name__ == '__main__':
    # main1()
    # main2()
    # main3()
    # main4()
    compare_times()
