""" Trains an agent with (stochastic) Policy Gradients on Cancer Invasion. """
import numpy as np
import pickle
from cancer_env import CancerEnv

# hyperparameters
H = 400 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 3 * 3 # input dimensionality: 3x3 grid nut
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = 0.1*np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = 0.1*np.random.randn(H,9) / np.sqrt(H) # 9 actions
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = model['W1'] @ x
  h[h<0] = 0 # ReLU nonlinearity
  logp = h @ model['W2']
  p = np.exp(logp) / np.exp(logp).sum()
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = eph.T @ epdlogp
  #dh = np.outer(epdlogp, model['W2'])
  dh = epdlogp @ model['W2'].T
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}


env = CancerEnv(19)
observation = env.reset()  # 3 by 3 neibnut
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
istep = 0
while True:
  istep += 1
  if render and istep % 2 == 0:
    env.render()
    istep = 0
  # preprocess the observation, set input to network to be difference image
  x = observation.ravel()

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = np.random.choice(range(9), p=aprob)  # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state

  y = np.zeros(9)
  y[action] = 1
  dlogps.append(y - aprob) # !grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done = env.pg_step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    #print(('ep %d: game finished, total reward: %f' % (episode_number, reward_sum)) + ('' if reward_sum < 11 else '!!!!'))
    episode_number += 1
    if episode_number in [1, 50, 100, 1000, 1500]:
      render = True
    else:
      render = False
    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)  # int????
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr = discounted_epr - np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # !modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('ep %d : resetting env. episode reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
    env.plot_extra(episode_number, running_reward)
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env