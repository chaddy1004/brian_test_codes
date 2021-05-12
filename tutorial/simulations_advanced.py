import brian2 as bs
from matplotlib import pyplot as plt

def main1():
    bs.start_scope()
    # Parameters
    area = 20000 * bs.umetre ** 2
    Cm = 1 * bs.ufarad * bs.cm ** -2 * area
    gl = 5e-5 * bs.siemens * bs.cm ** -2 * area
    El = -65 * bs.mV
    EK = -90 * bs.mV
    ENa = 50 * bs.mV
    g_na = 100 * bs.msiemens * bs.cm ** -2 * area
    g_kd = 30 * bs.msiemens * bs.cm ** -2 * area
    VT = -63 * bs.mV

    # HH stands for Hudgkin-Huxley
    eqs_HH = '''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    
    dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
        (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
        (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
        
    dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
        (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
        
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    
    I : amp
    '''

    group = bs.NeuronGroup(1, eqs_HH,
                        threshold='v > -40*mV',
                        refractory='v > -40*mV',
                        method='exponential_euler')
    group.v = El
    state_monitor = bs.StateMonitor(group,'v', record=True)
    spike_monitor = bs.SpikeMonitor(group, variables='v')
    plt.figure(figsize=(9,4))
    for l in range(5):
        group.I = bs.rand() * 50 * bs.nA
        bs.run(10 * bs.ms)
        bs.axvline(l * 10, ls='--', c='k')
    bs.axhline(El / bs.mV, ls='-', c='lightgray', lw=3)
    state_time = state_monitor.t
    spike_time = spike_monitor.t
    plt.plot(state_time / bs.ms, spike_monitor.v[0] / bs.mV, '-b')
    plt.plot(spike_time / bs.ms, spike_monitor.v / bs.mV, 'ob')
    plt.xlabel('Time (ms)')
    plt.ylabel('v (mV)')
    plt.show()

if __name__ == '__main__':
    main1()