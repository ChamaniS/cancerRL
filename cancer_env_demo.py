import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class ActionSpace(object):

    def __init__(self, actions):
        self.actions = actions
        self.num = len(actions)


class CancerEnv:

    def __init__(self, maxsub=19):
        self.maxsub = maxsub
        self.maxnut = 10
        self.size = (self.maxsub+1, self.maxsub+1)
        self.outreward = 10
        self.starvereward = -5
        self.nutforact = [1]*8
        self.nutforact.append(3)
        self.possible_step = [[-1,-1],[-1,0],[-1,1],[0,1],[0,-1],[1,0],[1,-1],[1,1]]
        self.drate = 0.1
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.imloc = self.ax1.imshow(np.zeros((maxsub + 1, maxsub + 1)),vmin=0,vmax=2)
        self.imnut = self.ax2.imshow(np.ones((maxsub+1,maxsub+1))*self.maxnut,vmin=0,vmax=self.maxnut)
        self.fig.colorbar(self.imnut, ax=self.ax2)
        self.ax1.set_title('cell locations')
        self.ax2.set_title('nutrient concentration')
        plt.tight_layout()
        #self.fig, self.ax = plt.subplots()

    def step(self, action):
        self.is_reset = False
        locsubs = self.loc.nonzero()
        reward = 0
        if np.size(locsubs[0]) == 0:
            self.reset()
        else:
            rndperm = np.random.permutation(np.size(locsubs[0]))

            for i in rndperm:
                xsub = locsubs[0][i]
                ysub = locsubs[1][i]
                nuteaten = min(action[xsub, ysub], self.nut[xsub, ysub])
                reward += nuteaten  # nutrient reward
                self.nut[xsub, ysub] -= nuteaten
                if nuteaten == action[xsub, ysub]:
                    wherefrom = np.array([xsub, ysub])
                    possible_step = [[-1,-1],[-1,0],[-1,1],[0,1],[0,-1],[1,0],[1,-1],[1,1]]
                    choice = np.random.choice(range(8))
                    rndstep = possible_step[choice]
                    whereto = wherefrom + rndstep
                    if max(whereto) > self.maxsub or min(whereto) < 0:  # step outside
                        self.loc[xsub, ysub] = 0
                        reward += self.outreward  # extra reward
                    elif self.loc[whereto[0], whereto[1]] == 0:  # step within
                        self.loc[whereto[0], whereto[1]] = 1
                        if action[xsub, ysub] == 1:  # migrate
                            self.loc[wherefrom[0], wherefrom[1]] = 0
                else:
                    self.loc[xsub, ysub] = np.random.choice([0, 1], 1)
        return self.loc, self.nut, reward, self.is_reset

    def single_step(self, cellloc, action):
        self.is_reset = False
        reward = 0
        nuteaten = min(action, self.nut[cellloc])
        #reward += nuteaten  # nutrient reward
        self.nut[cellloc] -= nuteaten
        if nuteaten == action:
            wherefrom = np.array(cellloc)
            possible_step = [[-1,-1],[-1,0],[-1,1],[0,1],[0,-1],[1,0],[1,-1],[1,1]]
            choice = np.random.choice(range(len(possible_step)))
            rndstep = possible_step[choice]
            whereto = wherefrom + rndstep
            if max(whereto) > self.maxsub or min(whereto) < 0:  # step outside
                self.loc[cellloc] = 0
                reward += self.outreward  # extra reward
            elif self.loc[whereto[0], whereto[1]] == 0:  # step within an empty cell
                self.loc[whereto[0], whereto[1]] = 1
                if action == 1:  # migrate
                    self.loc[wherefrom[0], wherefrom[1]] = 0
        else:
            self.loc[cellloc] = np.random.choice([0, 1], 1)  # chance to die
            reward += self.starvereward
        locsubs = self.loc.nonzero()
        lenlocsubs = len(locsubs[0])
        if lenlocsubs == 0:
            self.is_reset = True
            nextcell = None
        else:
            rndind = np.random.choice(range(lenlocsubs), 1)
            nextcell = (np.asscalar(locsubs[0][rndind]), np.asscalar(locsubs[1][rndind]))
        return nextcell, self.nut[nextcell], reward, self.is_reset

    def pg_step(self, action):
        self.is_reset = False
        reward = 0
        nuteaten = min(self.nutforact[action], self.nut[self.oploc])
        self.nut[self.oploc] -= nuteaten
        if nuteaten == self.nutforact[action]:
            wherefrom = np.array(self.oploc)
            if action < 8:  # migrate
                whereto = wherefrom + self.possible_step[action]
                if max(whereto) > self.maxsub or min(whereto) < 0:  # step outside
                    self.loc[self.oploc] = 0
                    reward += self.outreward  # extra reward
                elif self.loc[whereto[0], whereto[1]] == 0:  # step within an empty cell
                    self.loc[whereto[0], whereto[1]] = 1
                    self.loc[wherefrom[0], wherefrom[1]] = 0
            else:  # divide
                choice = np.random.choice(range(len(self.possible_step)))
                divto = wherefrom + self.possible_step[choice]
                reward += 5
                if max(divto) > self.maxsub or min(divto) < 0:  # step outside:
                    reward += self.outreward  # extra reward
                elif self.loc[divto[0], divto[1]] == 0:
                    self.loc[divto[0], divto[1]] = 2
        else:
            self.loc[self.oploc] = np.random.choice([0, 1], 1)  # chance to die
            reward += self.starvereward

        locsubs = self.loc.nonzero()
        lenlocsubs = len(locsubs[0])
        if lenlocsubs == 0:
            self.is_reset = True
            neibnut = None
        else:
            # nutrient diffusion
            dnut = ndimage.filters.laplace(self.nut, mode='reflect')
            self.nut = self.nut + dnut * (self.drate/lenlocsubs)
            #
            rndind = np.random.choice(range(lenlocsubs), 1)
            self.oploc = (np.asscalar(locsubs[0][rndind]), np.asscalar(locsubs[1][rndind]))
            (nextx, nexty) = self.oploc
            rowaug = np.ones((1, self.maxsub+1))*self.maxnut
            colaug = np.ones((self.maxsub+3, 1))*self.maxnut
            augmentednut = np.vstack((rowaug, self.nut, rowaug))
            augmentednut = np.hstack((colaug, augmentednut, colaug))
            neibnut = augmentednut[nextx:nextx+3, nexty:nexty+3]
        return neibnut, reward, self.is_reset  # return nut in a neighborhood

    def render(self):
        #self.ax1.imshow(self.loc)
        #self.ax2.imshow(self.nut)
        #self.ax1.cla()
        #self.ax2.cla()
        self.imloc.set_data(self.loc)
        self.imnut.set_data(self.nut)
        self.fig.canvas.draw()
        plt.pause(0.01)

    def single_reward(self):
        pass

    def reset(self):
        self.nstep = 0
        self.loc = np.zeros(self.size)
        xini = round(self.size[0]/2)
        yini = round(self.size[1]/2)
        self.loc[xini, yini] = 1
        self.oploc = (xini, yini)
        self.nut = np.ones(self.size)*self.maxnut
        self.is_reset = True
        #return self.loc, self.nut, (xini, yini)
        #self.im = self.ax.imshow(self.nut)
        return np.ones((3, 3))*self.maxnut

