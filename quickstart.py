
# coding: utf-8

# # ChainerRL Quickstart Guide
# 
# This is a quickstart guide for users who just want to try ChainerRL for the first time.
# 
# If you have not yet installed ChainerRL, run the command below to install it:
# ```
# pip install chainerrl
# ```
# 
# If you have already installed ChainerRL, let's begin!
# 
# First, you need to import necessary modules. The module name of ChainerRL is `chainerrl`. Let's import `gym` and `numpy` as well since they are used later.

# In[1]:

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np


# ChainerRL can be used for any problems if they are modeled as "environments". [OpenAI Gym](https://github.com/openai/gym) provides various kinds of benchmark environments and defines the common interface among them. ChainerRL uses a subset of the interface. Specifically, an environment must define its observation space and action space and have at least two methods: `reset` and `step`.
# 
# - `env.reset` will reset the environment to the initial state and return the initial observation.
# - `env.step` will execute a given action, move to the next state and return four values:
#   - a next observation
#   - a scalar reward
#   - a boolean value indicating whether the current state is terminal or not
#   - additional information
# - `env.render` will render the current state.
# 
# Let's try 'CartPole-v0', which is a classic control problem. You can see below that its observation space consists of four real numbers while its action space consists of two discrete actions.

# In[2]:

env = gym.make('CartPole-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

obs = env.reset()
env.render()
print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)


# Now you have defined your environment. Next, you need to define an agent, which will learn through interactions with the environment.
# 
# ChainerRL provides various agents, each of which implements a deep reinforcement learning algorithm.
# 
# To use [DQN (Deep Q-Network)](http://dx.doi.org/10.1038/nature14236), you need to define a Q-function that receives an observation and returns an expected future return for each action the agent can take. In ChainerRL, you can define your Q-function as `chainer.Link` as below. Note that the outputs are wrapped by `chainerrl.action_value.DiscreteActionValue`, which implements `chainerrl.action_value.ActionValue`. By wrapping the outputs of Q-functions, ChainerRL can treat discrete-action Q-functions like this and [NAFs (Normalized Advantage Functions)](https://arxiv.org/abs/1603.00748) in the same way.

# In[3]:

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)


# If you want to use CUDA for computation, as usual as in Chainer, call `to_gpu`.

# In[4]:

# Uncomment to use CUDA
# q_func.to_gpu(0)


# You can also use ChainerRL's predefined Q-functions.

# In[5]:

_q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    obs_size, n_actions,
    n_hidden_layers=2, n_hidden_channels=50)


# As in Chainer, `chainer.Optimizer` is used to update models.

# In[6]:

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(_q_func)


# A Q-function and its optimizer are used by a DQN agent. To create a DQN agent, you need to specify a bit more parameters and configurations.

# In[7]:

# Set the discount factor that discounts future rewards.
gamma = 0.95

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
    _q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_frequency=1,
    target_update_frequency=100, phi=phi)


# Now you have an agent and an environment. It's time to start reinforcement learning!
# 
# In training, use `agent.act_and_train` to select exploratory actions. `agent.stop_episode_and_train` must be called after finishing an episode. You can get training statistics of the agent via `agent.get_statistics`.

# In[11]:

n_episodes = 200
max_episode_len = 200
for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:
        # Uncomment to watch the behaviour
        # env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done)
print('Finished.')


# Now you finished training the agent. How good is the agent now? You can test it by using `agent.act` and `agent.stop_episode` instead. Exploration such as epsilon-greedy is not used anymore.

# In[12]:

for i in range(10):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        env.render()
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    agent.stop_episode()


# If test scores are good enough, the only remaining task is to save the agent so that you can reuse it. What you need to do is to simply call `agent.save` to save the agent, then `agent.load` to load the saved agent.

# In[10]:

# Save an agent to the 'agent' directory
agent.save('agent')

# Uncomment to load an agent from the 'agent' directory
# agent.load('agent')


# RL completed!
# 
# But writing code like this every time you use RL might be boring. So, ChainerRL has utility functions that do these things.

# In[22]:

# Set up the logger to print info messages for understandability.
import logging
import sys
gym.undo_logger_setup()  # Turn off gym's default logger settings
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

chainerrl.experiments.train_agent_with_evaluation(
    agent, env,
    steps=2000,           # Train the agent for 2000 steps
    eval_n_runs=10,       # 10 episodes are sampled for each evaluation
    max_episode_len=200,  # Maximum length of each episodes
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='result')      # Save everything to 'result' directory


# That's all of the ChainerRL quickstart guide. To know more about ChainerRL, please look into the `examples` directory and read and run the examples. Thank you!
