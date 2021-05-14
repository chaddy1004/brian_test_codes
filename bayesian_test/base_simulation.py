import abc
from matplotlib import pyplot as plt
import brian2 as bs


class BaseSimulation():
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def visualise_connectivity(self, S):
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

    @abc.abstractmethod
    def define_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def run_simulation(self, time):
        raise NotImplementedError
