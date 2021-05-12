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
    # adding different weights per synapse
    # ex0 distance-dependent connectivity function=
    # important for a lot neurons that have weaker inhibitory/excitatory connections as the distances get wider
    bs.start_scope()
    n_neurons = 30
    neuron_spacing = 50 * bs.umetre
    width = n_neurons / 4.0 * neuron_spacing

    G = bs.NeuronGroup(n_neurons, 'x:metre')
    G.x = 'i*neuron_spacing'

    # All synapses are connected (excluding self-connections)
    S = bs.Synapses(G, G, 'w:1')
    S.connect(condition='i!=j')

    # basically, any variable you use in the definition of equations is usable as an actual variable in code and vice versa
    # therefore, even if the variable width is labelled as not being used, it actually is lol
    S.w = 'exp(-(x_pre-x_post)**2/(2*width**2))'

    # visualise_connectivity(S)
    plt.clf()

    plt.scatter(S.x_pre / bs.um, S.x_post / bs.um, S.w * 20)
    plt.xlabel('source neuron position (um)')
    plt.ylabel('Target neuron position (um)')
    plt.show()

def main2():
    # STDP curve
    tau_pre = 20*bs.ms
    tau_post = 20*bs.ms

    A_pre = 0.01
    A_post = -A_pre * 1.05
    delta_t = bs.linspace(-50, 50, 100)*bs.ms

    W = bs.where(delta_t > 0, A_pre*bs.exp(-delta_t/tau_pre), A_post*bs.exp(delta_t/tau_post))
    plt.plot(delta_t/bs.ms, W)
    plt.xlabel(r'$\Delta t$ (ms)')
    plt.ylabel('W')
    plt.axhline(0, ls='-', c='k') # add horizontal line across the axis
    plt.show()

def main3():
    # STDP with eligibility trace

    bs.start_scope()

    tau_pre = 20*bs.ms
    tau_post = 20*bs.ms

    w_max = 0.01
    A_pre = 0.01
    A_post = -A_pre*tau_pre/tau_post*1.05

    n_neurons = 2

    G = bs.NeuronGroup(n_neurons, 'v:1', threshold='t>(1+i)*10*ms', refractory=100*bs.ms)

    # (event driven) syntax means that Brian should only update the equation at the time of an event (a spike)
    model = """
    w: 1
    dapre/dt = -apre/tau_pre : 1 
    dapost/dt = -apost/tau_post : 1 
    """

    on_pre_model = """
    v_post += w
    apre += A_pre
    w = clip(w + apost, 0, w_max)
    """

    on_post_model = """
    apost += A_post
    w = clip(w+apre, 0, w_max)
    """
    # by clipping, we keep the weight between 0 and w_max

    S = bs.Synapses(source=G, target=G, model=model, on_pre=on_pre_model, on_post=on_post_model, method='euler')

    S.connect(j="k for k in range(0,n_neurons-1)")

    state_monitor = bs.StateMonitor(source=S,variables= ['w', 'apre', 'apost'], record =True)

    visualise_connectivity(S)

    bs.run(30*bs.ms)

    time = state_monitor.t
    apre_0 = state_monitor.apre[0]
    apost_0 = state_monitor.apost[0]

    apre_1 = state_monitor.apre[1]
    apost_1 = state_monitor.apost[1]

    w_0 = state_monitor.w[0]
    w_1 = state_monitor.w[1]



    plt.figure(figsize=(4, 8))
    plt.figure(figsize=(4, 8))
    plt.subplot(211)
    plt.plot(time / bs.ms, apre_0, label='apre_0')
    plt.plot(time / bs.ms, apost_0, label='apost_0')

    plt.plot(time / bs.ms, apre_1, label='apre_1')
    plt.plot(time / bs.ms, apost_1, label='apost_1')

    plt.legend()
    plt.subplot(212)
    plt.plot(time / bs.ms, w_0, label='w[0]')
    plt.plot(time / bs.ms, w_1, label='w[1]')
    plt.legend(loc='best')
    plt.xlabel('Time (ms)')
    plt.show()








if __name__ == '__main__':
    # main1()
    # main2()
    main3()