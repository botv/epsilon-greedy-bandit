import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit
from agents import Epsilon_Greedy


class Session:
    def __init__(self, accuracy, arms, steps):
        self.steps = steps
        self.arms_array = np.flip(np.sort(np.random.rand(accuracy,
                                                         arms), 1), 1)

    def pull(self, q, epsilon):
        score = 0
        results = []
        for arms in self.arms_array:
            local = []
            bandit = Bandit(arms)
            agent = Epsilon_Greedy(len(arms), q)
            for k in range(self.steps):
                action = agent.epsilon_greedy(bandit, epsilon)
                reward = bandit.reward(action)
                agent.steps[action] += 1
                agent.update_estimates(action, reward, k)
                local.append(action)
                score += reward
            results.append(local)
        print(str(score / len(self.arms_array)) + " / " + str(self.steps))
        return(results)

    def graph_percent_optimal_action(self, actions, labels):
        def fix_axes(actions):
            plot = []
            steps = np.swapaxes(actions, 0, 1)
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
