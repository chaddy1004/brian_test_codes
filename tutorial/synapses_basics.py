import brian2 as bs
from matplotlib import pyplot as plt


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


def main1():
    #  checking effect of synaptic connectipn
    bs.start_scope()
    n_neurons = 2
    eqs = '''
    dv/dt = (I-v)/tau : 1 (unless refractory)
    I : 1
    tau: second
    '''
    threshold = 'v>1'
    reset = 'v = 0'
    refractory = 10 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=eqs, threshold=threshold, reset=reset, method='exact', refractory=refractory)

    G.I = [2, 0]  # represents the I in eqs
    # driving current should be bigger than the threshold, otherwise it wont spike at all

    # I = 0 means that the voltage wont change since there is no driving current
    G.tau = [10, 100] * bs.ms  # represents the tau in eqs
    # unlike last time, the tau is defined with the neuron so we see the effects of different values

    # So in total, what this model says is that whenever two neurons in G are connected by a synapse,
    # when the source neuron fires a spike the target neuron will have its value of v increased by 0.2.
    S = bs.Synapses(source=G, target=G, on_pre='v_post +=  0.2')
    # calling just S.connect() without any parameters just connects every source neuron with every target neuron
    # fully connected
    S.connect(i=0, j=1)
    visualise_connectivity(S)
    state_monitor = bs.StateMonitor(G, 'v', record=True)
    bs.run(100 * bs.ms)

    plt.plot(state_monitor.t / bs.ms, state_monitor.v[0], label='Neuron 0')
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[1], label='Neuron 1')
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.legend()
    plt.show()


def main2():
    # setting synaptic weight
    bs.start_scope()
    n_neurons = 3
    eqs = '''
    dv/dt = (I-v)/tau : 1 (unless refractory)
    I : 1
    tau: second
    '''
    threshold = 'v>1'
    reset = 'v = 0'
    refractory = 10 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=eqs, threshold=threshold, reset=reset, method='exact', refractory=refractory)

    G.I = [2, 0, 0]  # represents the I in eqs
    # driving current should be bigger than the threshold, otherwise it wont spike at all

    # I = 0 means that the voltage wont change since there is no driving current
    G.tau = [10, 100, 100] * bs.ms  # represents the tau in eqs
    # unlike last time, the tau is defined with the neuron so we see the effects of different values

    #     model : `str`, `Equations`, optional
    #         The model equations for the synapses.
    # you need to set this as model= in Synapses in order to incorporate weight
    synapse_model = 'w : 1'

    # So in total, what this model says is that whenever two neurons in G are connected by a synapse,
    # when the source neuron fires a spike the target neuron will have its value of v increased by 0.2.
    S = bs.Synapses(source=G, target=G, model=synapse_model, on_pre='v_post +=  w')
    S.connect(i=0, j=[1, 2])
    # So this will give a synaptic connection from 0 to 1 with weight 0.2=0.2*1 and from 0 to 2 with weight 0.4=0.2*2.
    S.w = 'j*0.2'

    state_monitor = bs.StateMonitor(G, 'v', record=True)
    bs.run(100 * bs.ms)

    plt.plot(state_monitor.t / bs.ms, state_monitor.v[0], label='Neuron 0')
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[1], label='Neuron 1')
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[2], label='Neuron 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.legend()
    plt.show()


def main3():
    # introducing delay
    bs.start_scope()
    n_neurons = 3
    eqs = '''
    dv/dt = (I-v)/tau : 1 (unless refractory)
    I : 1
    tau: second
    '''
    threshold = 'v>1'
    reset = 'v = 0'
    refractory = 10 * bs.ms
    G = bs.NeuronGroup(N=n_neurons, model=eqs, threshold=threshold, reset=reset, method='exact', refractory=refractory)

    G.I = [2, 0, 0]  # represents the I in eqs
    # driving current should be bigger than the threshold, otherwise it wont spike at all

    # I = 0 means that the voltage wont change since there is no driving current
    G.tau = [10, 100, 100] * bs.ms  # represents the tau in eqs
    # unlike last time, the tau is defined with the neuron so we see the effects of different values

    synapse_model = 'w : 1'

    # So in total, what this model says is that whenever two neurons in G are connected by a synapse,
    # when the source neuron fires a spike the target neuron will have its value of v increased by 0.2.
    S = bs.Synapses(source=G, target=G, model=synapse_model, on_pre='v_post +=  w')
    S.connect(i=0, j=[1, 2])
    S.w = 'j*0.2'
    S.delay = 'j*2*ms'

    state_monitor = bs.StateMonitor(G, 'v', record=True)
    bs.run(100 * bs.ms)

    plt.plot(state_monitor.t / bs.ms, state_monitor.v[0], label='Neuron 0')
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[1], label='Neuron 1')
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[2], label='Neuron 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.legend()
    plt.show()


def main4():
    # more complex connectivity
    bs.start_scope()
    n_neurons = 3
    tau = 100 * bs.ms
    eqs = '''
    dv/dt = (I-v)/tau : 1 (unless refractory)
    I : 1
    '''
    threshold = 'v>1'
    reset = 'v = 0'
    refractory = 10 * bs.ms

    G = bs.NeuronGroup(N=n_neurons, model=eqs, threshold=threshold, reset=reset, method='exact', refractory=refractory)

    G.I = [2, 0, 0]
    synapse_model = 'w : 1'

    #         p : float, str, optional
    #             The probability to create ``n`` synapses wherever the ``condition``
    #             evaluates to true. Cannot be used with generator syntax for ``j``.
    S = bs.Synapses(source=G, target=G, model=synapse_model, on_pre='v_post +=  w')
    S.connect(condition='i!=j', p=0.5)
    S.w = 'j*0.2'

    state_monitor = bs.StateMonitor(G, 'v', record=True)
    bs.run(100 * bs.ms)

    plt.plot(state_monitor.t / bs.ms, state_monitor.v[0], label='Neuron 0')
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[1], label='Neuron 1')
    plt.plot(state_monitor.t / bs.ms, state_monitor.v[2], label='Neuron 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('v')
    plt.legend()
    plt.show()

    visualise_connectivity(S=S)


def main5():
    # the effect of probability in connections
    bs.start_scope()

    n_neurons = 10
    G = bs.NeuronGroup(n_neurons, model='v:1')

    for p in [0.1, 0.5, 1.0]:
        S = bs.Synapses(G, G)
        # connect neurons that do not have the same index with probability p
        S.connect(condition='i!=j', p=p)
        visualise_connectivity(S)
        plt.suptitle('p = ' + str(p))


# Feel like it would be important for making the pose cell in ratslam
# It had local connections
def main6():
    # connecting only the neighbouring neurons
    bs.start_scope()

    n_neurons = 10
    G = bs.NeuronGroup(n_neurons, 'v:1')

    S = bs.Synapses(G, G)
    # connect only if the neuron is less than 4 spaces and the neuron index isnt the same
    S.connect(condition='abs(i-j)<4 and i!=j')
    visualise_connectivity(S)


def main7():
    # using generator syntax to create connections
    bs.start_scope()

    n_neurons = 10
    G = bs.NeuronGroup(n_neurons, 'v:1')

    S = bs.Synapses(G, G)
    # connect only if the neuron is less than 4 spaces and the neuron index isnt the same
    # skip in invalid is needed here since on the edge connections, the i+4 goes over bound and causes error
    S.connect(j='k for k in range(i-3, i+4) if i != k', skip_if_invalid=True)
    visualise_connectivity(S)


def main8():
    # using generator syntax to create connections
    bs.start_scope()

    n_neurons = 10
    G = bs.NeuronGroup(n_neurons, 'v:1')

    S = bs.Synapses(G, G)
    """
            i : int, ndarray of int, optional
            The presynaptic neuron indices (in the form of an index or an array
            of indices). Must be combined with ``j`` argument.
        j : int, ndarray of int, str, optional
            The postsynaptic neuron indices. It can be an index or array of
            indices if combined with the ``i`` argument, or it can be a string
            generator expression.
    """
    # the above is the reason why j="i" works but not i = "j"
    # only j can take in string. i has to take in int, or ndarray of int
    S.connect(j='i', skip_if_invalid=True)

    # You can also do it the following way
    # S.connect(condition = 'i==j', skip_if_invalid=True)

    visualise_connectivity(S)


if __name__ == '__main__':
    # main1()
    # main2()
    # main3()
    # main4()
    # main5()
    # main6()
    # main7()
    main8()
