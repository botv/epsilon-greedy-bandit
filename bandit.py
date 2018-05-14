import ast
import sys

import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    def __init__(self, arms):
        self.arms = arms

    def reward(self, a):
        return(np.random.rand() < self.arms[a])


class Agent:
    def __init__(self, arms, q=0.5):
        self.estimates = np.full(arms, q)
        self.steps = {}
        for arm in range(arms):
            self.steps[arm] = 0

    def epsilon_greedy(self, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(np.arange(len(self.estimates)))
        else:
            action = np.argmax(self.estimates)
        return(action)

    def update_estimates(self, a, r):
        self.estimates[a] += (1 / (self.steps[a] + 1)
                              * (r - self.estimates[a]))


class Session:
    def __init__(self, accuracy, arms, steps):
        self.steps = steps
        self.arms_array = np.flip(np.sort(np.random.rand(accuracy,
                                                         arms), 1), 1)

    def run_single(self, q, epsilon):
        score = 0
        results = []
        for arms in self.arms_array:
            local = []
            bandit = Bandit(arms)
            agent = Agent(len(arms), q)
            for k in range(self.steps):
                action = agent.epsilon_greedy(epsilon)
                reward = bandit.reward(action)
                agent.steps[action] += 1
                agent.update_estimates(action, reward)
                local.append(action)
                score += reward
            results.append(local)
        print(str(score / len(self.arms_array)) + " / " + str(self.steps))
        return(results)

    def graph_percent_optimal_action(self, actions, labels):
        def fix_axes(action_list):
            plot = []
            steps = np.swapaxes(action_list, 0, 1)
            for a in steps:
                plot.append(100 * (np.count_nonzero(a == 0) / len(a)))
            return plot
        for index, arr in enumerate(actions):
            plt.plot(fix_axes(arr), label=labels[index])
        plt.axis([0, self.steps, 0, 100])
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('% Optimal Action')
        plt.show()

    def run(self, agents):
        results = []
        labels = []
        for agent in agents:
            labels.append(r'$Q_0 = %s, \epsilon = %s$'
                          % (agent[0], agent[1]))
            print("Q%s = %s, %s = %s"
                  % (chr(8320), agent[0], chr(949), agent[1]))
            results.append(self.run_single(agent[0], agent[1]))
        self.graph_percent_optimal_action(results, labels)


def main():
    session = Session(1000, 2, 500)
    try:
        session.run(ast.literal_eval(sys.argv[1]))
    except IndexError:
        print("No agents were specified.")


if __name__ == '__main__':
    # Refactoring
    main()
