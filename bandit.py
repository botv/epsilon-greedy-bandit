import numpy as np


class Bandit:
    def __init__(self, arms):
        self.arms = arms

    def reward(self, a):
        return(np.random.rand() < self.arms[a])
