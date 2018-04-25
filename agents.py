import numpy as np


class Epsilon_Greedy:
    def __init__(self, arms, q=0.5):
        self.estimates = np.full(arms, q)
        self.steps = {}
        for arm in range(arms):
            self.steps[arm] = 0

    def epsilon_greedy(self, bandit, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(np.arange(len(self.estimates)))
        else:
            action = np.argmax(self.estimates)
        return(action)

    def update_estimates(self, a, r, k):
        self.estimates[a] += ((1 / (self.steps[a] + 1)
                               * (r - self.estimates[a])))


class Softmax:
    def __init__(self, arms, q=0.5):
        self.estimates = np.full(arms, q)
        self.steps = {}
        for arm in range(arms):
            self.steps[arm] = 0

    def softmax(self, bandit, k):
        temperature = 1 / ((k * 0.3) + 1)
        pi = (np.exp(np.divide(self.estimates, temperature))
              / np.sum(np.exp(np.divide(self.estimates, temperature))))
        action = np.random.choice(len(bandit.arms), p=pi)
        return(action)

    def update_estimates(self, a, r, k):
        self.estimates[a] += ((1 / (self.steps[a] + 1)
                               * (r - self.estimates[a])))
