import brian2 as bs
from matplotlib import pyplot as plt

# https://brian2.readthedocs.io/en/stable/resources/tutorials/1-intro-to-brian-neurons.html
def main1():
    bs.start_scope()
    tau = 10 * bs.ms
    # equations must end with : unit
    # unit is the SI unit of that variable
    # the unit is 1 since the number "1" is unitless
    # v represents voltage, but we just keep it unitless for simplicity
    # 1/s not part of the unit since the unit represents the unit of the variable itself
    # rather than the unit of the equation
    eqs = '''
    dv/dt = (1-v)/tau: 1
    '''

    G = bs.NeuronGroup(1, model=eqs, method='exact')
    # record : bool, sequence of ints
    #     Which indices to record, nothing is recorded for ``False``,
    #     everything is recorded for ``True`` (warning: may use a great deal of
    #     memory), or a specified subset of indices.
    M = bs.StateMonitor(G, 'v', record=0)
    print('Before v = %s' % G.v[0])
    bs.run(100 * bs.ms)  # runs the simulation for 100ms
    print('After v = %s' % G.v[0])

    plt.plot(M.t / bs.ms, M.v[0])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()


def main2():
    tau = 10 * bs.ms
    eqs = '''
    dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1
    '''
    G = bs.NeuronGroup(1, eqs, method="euler")
    G.v = 5  # we can set the initial value
    M = bs.StateMonitor(G, 'v', record=0)

    bs.run(60 * bs.ms)

    plt.plot(M.t / bs.ms, M.v[0])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()


def main3():
    # Adding Spikes
    bs.start_scope()

    tau = 10 * bs.ms
    eqs = '''
    dv/dt = (1-v)/tau : 1
    '''

    # conditions for spiking models
    threshold = 'v>0.8'
    reset = 'v = -0.8'
    G = bs.NeuronGroup(1, eqs, threshold=threshold, reset=reset, method='exact')

    M = bs.StateMonitor(G, 'v', record=0)
    bs.run(50 * bs.ms)
    plt.plot(M.t / bs.ms, M.v[0])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()

    # you can also add spike monitor
    spike_monitor = bs.SpikeMonitor(G)

    bs.run(50 * bs.ms)

    print(f"Spike Times: {spike_monitor.t[:]}")


def main4():
    # Incorporation of refractory period
    bs.start_scope()

    tau = 10 * bs.ms
    # the (unless refractory) is necessary
    # refer to the documentation for more detail
    eqs = '''
    dv/dt = (1-v)/tau : 1 (unless refractory)
    '''
    equation = bs.Equations(eqs)
    # conditions for spiking models
    threshold = 'v>0.8'
    reset = 'v = -0.8'
    refractory = 5 * bs.ms
    G = bs.NeuronGroup(1, eqs, threshold=threshold, reset=reset, method='exact', refractory=refractory)

    state_monitor = bs.StateMonitor(G, 'v', record=0)
    spike_monitor = bs.SpikeMonitor(G)

    bs.run(50 * bs.ms)
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[0])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()


def main5():
    # multiple neurons
    n_neurons = 100
    tau = 10 * bs.ms
    eqs = """
    dv/dt = (2-v)/tau: 1
    """
    # conditions for spiking models
    threshold = 'v>1.8'
    reset = 'v = -0.8'
    refractory = 5 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=eqs, threshold=threshold, reset=reset, method='exact')
    G.v = 0
    # you can also set up certain neurons individually
    # G.v[0] = 'rand()'
    # G.v[10] = 'rand()'
    # G.v[50] = 'rand()'
    # G.v[99] = 'rand()'
    G.v = 'rand()'

    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)
    bs.run(50 * bs.ms)

    # spike
    plt.plot(spike_monitor.t / bs.ms, spike_monitor.i, 'pk')  # the .k -> p means point marker, k means black
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()

    plt.clf()
    # You can also plot the actual voltage change of each neuron if you recorded it with state monitors
    print(G.v[0])
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[0])
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[10])
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[50])
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[98])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()


def main6():
    # neurons with parameters
    n_neurons = 100
    tau = 10 * bs.ms
    v0_max = 3.0
    duration = 1000 * bs.ms
    # v0: 1 declares a new per-neuron parameter v0
    eqs = """
    dv/dt = (v0-v)/tau: 1 (unless refractory)
    v0 : 1
    """

    # conditions for spiking models
    threshold = 'v>1'
    reset = 'v = 0'
    refractory = 5 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=eqs, threshold=threshold, reset=reset, method='exact', refractory=refractory)

    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)
    """
    The line G.v0 = 'i*v0_max/(N-1)' initialises the value of v0 for each neuron varying from 0 up to v0_max. 
    The symbol i when it appears in strings like this refers to the neuron index.
    """
    G.v0 = 'i*v0_max/(N-1)'
    bs.run(duration=duration)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(spike_monitor.t / bs.ms, spike_monitor.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.subplot(122)
    plt.plot(G.v0, spike_monitor.count / duration)
    plt.xlabel('v0')
    plt.ylabel('Firing rate (sp/s)')
    plt.show()

    plt.clf()
    # plt.plot(state_monitor.t / bs.ms, state_monitor.v[0])
    # plt.plot(state_monitor.t / bs.ms, state_monitor.v[10])
    # plt.plot(state_monitor.t / bs.ms, state_monitor.v[50])
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[30])
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[40])
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[70])
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.show()

def main7():
    # stochastic neurons
    # multiple neurons
    n_neurons = 100
    tau = 10 * bs.ms
    v0_max = 3.0
    duration = 1000 * bs.ms
    sigma = 0.2
    # v0: 1 declares a new per-neuron parameter v0
    # n Brian, we can do this by using the symbol xi in differential equations.
    # Strictly speaking, this symbol is a “stochastic differential” but you can sort of thinking of it as just a
    # Gaussian random variable with mean 0 and standard deviation 1.
    eqs = '''
    dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
    v0 : 1
    '''

    # conditions for spiking models
    threshold = 'v>1'
    reset = 'v = 0'
    refractory = 5 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=eqs, threshold=threshold, reset=reset, method='euler', refractory=refractory)

    state_monitor = bs.StateMonitor(G, 'v', record=True)
    spike_monitor = bs.SpikeMonitor(G)
    """
    The line G.v0 = 'i*v0_max/(N-1)' initialises the value of v0 for each neuron varying from 0 up to v0_max. 
    The symbol i when it appears in strings like this refers to the neuron index.
    """
    G.v0 = 'i*v0_max/(N-1)'
    bs.run(duration=duration)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(spike_monitor.t / bs.ms, spike_monitor.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.subplot(122)
    plt.plot(G.v0, spike_monitor.count / duration)
    plt.xlabel('v0')
    plt.ylabel('Firing rate (sp/s)')
    plt.show()


if __name__ == '__main__':
    # main1()
    # main2()
    # main3()
    # main4()
    # main5()
    main6()
    # main7()
