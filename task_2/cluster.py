from numpy import mean, array, random
from random import choice
from functools import reduce
from numpy import linalg, transpose, zeros, identity

EPSILON = 0.01


class Cluster:
    def __init__(self, probability):
        self.old_mu = array([0., 0.])
        self.old_sigma = array([[0., 0.], [0., 0.]])
        self.pi = probability
        self.mu = array([random.uniform(1, 5), random.uniform(1, 5)])
        self.sigma = array([[random.uniform(0, 9), 0],
                            [0, random.uniform(0, 9)]])
        while(linalg.det(self.sigma) < 0.5):
            self.sigma = array([[random.uniform(0, 9), 0],
                                [0, random.uniform(0, 9)]])

    def update_mu(self, new_mu):
        self.old_mu = self.mu[:]
        self.mu = new_mu

    def update_sigma(self, new_sigma):
        self.old_sigma = self.sigma[:]
        self.sigma = new_sigma

    def check_converge(self):
        dif = self.old_mu - self.mu
        return (dif[0] < EPSILON and dif[1] < EPSILON)
