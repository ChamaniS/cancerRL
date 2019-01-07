import numpy as np
import matplotlib.pyplot as plt
from cancer_env import CancerEnv
import matplotlib.gridspec as gridspec


class Experiment(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3 = self.fig.add_subplot(3, 1, 3)

    def run_rc_agent(self, max_number_of_episodes=1, interactive=False):
        self.episode_reward = np.zeros(max_number_of_episodes)
        for episode_number in range(max_number_of_episodes):
            loc, nut, _ = self.env.reset()
            done = False
            t = 0
            accu_reward = 0
            while not done:
                t += 1
                action = self.agent.act(loc, nut)
                loc, nut, reward, done = self.env.step(action)
                accu_reward += reward
                if interactive:
                    self.update_display_step()

            self.episode_reward[episode_number] = accu_reward
            if interactive:
                self.ax3.clear()
                self.ax3.plot(range(episode_number+1), self.episode_reward[:episode_number+1])
        if not interactive:
            self.fig.clf()
            plt.plot(range(max_number_of_episodes), self.episode_reward)
            plt.show()

    def run_single_agent(self, max_number_of_episodes, interactive=False):
        self.episode_reward = np.zeros(max_number_of_episodes)
        for episode_number in range(max_number_of_episodes):
            _, _, cellloc = self.env.reset()
            done = False
            t = 0
            accu_reward = 0
            while not done:
                t += 1
                action = self.agent.act()
                cellloc, _, reward, done = self.env.single_step(cellloc, action)
                accu_reward += reward
                if interactive:
                    self.update_display_step()

            self.episode_reward[episode_number] = accu_reward
            if interactive:
                self.ax3.clear()
                self.ax3.plot(range(episode_number+1), self.episode_reward[:episode_number+1])
        if not interactive:
            self.fig.clf()
            plt.plot(range(max_number_of_episodes), self.episode_reward)
            plt.show()
        return sum(self.episode_reward) / max_number_of_episodes

    def run_qlin_agent(self, max_number_of_episodes, interactive=False):
        self.episode_reward = np.zeros(max_number_of_episodes)
        for episode_number in range(max_number_of_episodes):
            _, nutfield, cellloc = self.env.reset()
            cellnut = nutfield[cellloc]
            done = False
            t = 0
            accu_reward = 0
            while not done:
                t += 1
                action = self.agent.act(cellnut)
                cellloc, next_cellnut, reward, done = self.env.single_step(cellloc, action)
                self.agent.learn(cellnut, action, reward, next_cellnut, done)
                cellnut = next_cellnut
                accu_reward += reward
                if interactive:
                    self.update_display_step()

            self.episode_reward[episode_number] = accu_reward
            if interactive:
                self.ax3.clear()
                self.ax3.plot(range(episode_number+1), self.episode_reward[:episode_number+1])
        if not interactive:
            self.fig.clf()
            plt.plot(range(max_number_of_episodes), self.episode_reward)
            plt.show()
        return sum(self.episode_reward)/max_number_of_episodes, self.agent.theta


    def update_display_step(self):
        self.ax1.imshow(self.env.loc)
        self.ax2.imshow(self.env.nut)
        plt.pause(0.05)


class RandomCentralAgent(object):
    def __init__(self):
        pass

    def act(self, loc, nut):
        action = np.zeros(loc.shape)
        locsubs = loc.nonzero()
        action[locsubs] = np.random.choice([1, 3], len(locsubs[0]))

        #action = loc * np.random.choice([1, 3], 1)
        return action


class QLinSingleAgent:
    def __init__(self, actions=[1, 3], ntheta=22, epsilon=0.01, alpha=0.1, gamma=1):
        self.theta = np.zeros(ntheta)
        # learning parameters
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # actions
        self.actions = actions  # nutrients needed for each action
        self.num_actions = len(actions)

    def feature_extractor(self, nut, action):
        nuttile = np.zeros(11)
        nuttile[int(round(nut))] = 1
        actionindex = np.zeros(self.num_actions, dtype=np.int)
        ind = 0 if action == 1 else 1  # hardcoded crap
        actionindex[ind] = 1
        feature = np.concatenate([actionindex[i] * nuttile for i in range(self.num_actions)])
        return feature

    def act(self, nut):
        ## epsilon greedy policy
        if np.random.random() < self.epsilon:
            i = np.random.randint(0, len(self.actions))
        else:
            q = []  # replace 0 with the correct calculation here
            for a in self.actions:
                # print(a)
                q.append(np.sum(self.theta.transpose() * self.feature_extractor(nut, a)))
            if q.count(max(q)) > 1:
                best = [i for i in range(len(self.actions)) if q[i] == max(q)]
                i = np.random.choice(best)
            else:
                i = q.index(max(q))

        action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2, done):

        """
        Q-learning with FA
        theta <- theta + alpha * td_delta * f(s,a)
        where
        td_delta = reward + gamma * max(Q(s') - Q(s,a))
        Q(s,a) = thetas * f(s,a)
        max(Q(s')) = max( [ thetas * f(s'a) for a in all actions] )

        """
        ## Implement the q-learning update here
        if not done:
            q = []
            for a in self.actions:
                q.append(np.sum(self.theta.transpose() * self.feature_extractor(state2, a)))
            maxqnew = max(q)  # replace 0 with the correct calculation
            oldv = np.sum(self.theta.transpose() * self.feature_extractor(state1, action1))  # replace 0 with the correct calculation
            td_target = reward + self.gamma * maxqnew
            td_delta = td_target - oldv
            self.theta += self.alpha * td_delta * self.feature_extractor(state1, action1).transpose()  # replace 0 with the correct calculation


class RandomSingleAgent(object):
    def __init__(self):
        pass

    def act(self):
        action = np.random.choice([1, 3])
        return action


maxsub = 19
env = CancerEnv(maxsub)
#rc_agent = RandomCentralAgent()
#experiment = Experiment(env, agent)
#experiment.run_rc_agent(10, True)

#
# agent = RandomSingleAgent()
# experiment = Experiment(env, agent)
# avgr = experiment.run_single_agent(1000, False)
# print(avgr)

agent = QLinSingleAgent()
experiment = Experiment(env, agent)
avgr, theta = experiment.run_qlin_agent(500, False)
print(avgr)
print(theta)