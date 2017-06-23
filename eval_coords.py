import subprocess
import os
import gym
# import cv2
import time
import numpy as np
import json
import signal
import pickle

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

linux = True

game_processes = []

# port render_freq msg_freq server
if linux:
    game_processes.append(
        subprocess.Popen("./game_linux.x86_64 5001 200 100 1 a", shell=True, stdout=subprocess.PIPE,
                         preexec_fn=os.setsid))
else:
    game_processes.append(
        subprocess.Popen("open -a game_mac.app --args 5000 10 10 1 aaaaaaaaaa", shell=True, stdout=subprocess.PIPE,
                         preexec_fn=os.setsid))

time.sleep(7)

game = gym.make('Unity-v0')
game.configure("5001")

def signal_handler(signal, frame):
    print("killing game processes...")
    for pro in game_processes:
        try:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            pro.kill()
        except:
            pass

# doesn't always work somehow
signal.signal(signal.SIGINT, signal_handler)

def get_extra(obs):
    data = bytearray(obs["extra"]).decode("utf-8")
    obj = json.loads(data)

    return obj

def get_coords(extra):
    coords = np.array(extra['coords'])
    ret = coords
    return ret

def agent_act(a_i):
    power = 14
    power_skew = power / np.sqrt(2)
    my_vec = [0,0]

    if a_i == 0:
        my_vec = [power,0]
    elif a_i == 1:
        my_vec = [-power,0]
    elif a_i == 2:
        my_vec = [0,power]
    elif a_i == 3:
        my_vec = [0,-power]
    elif a_i == 4:
        my_vec = [power_skew, power_skew]
    elif a_i == 5:
        my_vec = [power_skew, -power_skew]
    elif a_i == 6:
        my_vec = [-power_skew, power_skew]
    elif a_i == 7:
        my_vec = [-power_skew, -power_skew]
    return my_vec

def random_action_func():
    return np.random.randint(8)

q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(2, 8, n_hidden_layers=2, n_hidden_channels=50)
q_func.to_gpu(0)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
gamma = 0.95
explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=random_action_func)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)

agent = chainerrl.agents.DoubleDQN(q_func, optimizer, replay_buffer, gamma, explorer, minibatch_size = 4, replay_start_size = 500, update_interval = 1, target_update_interval = 100, phi=phi)
agent.load("agent_coords")

reward = 0
evals = {"rewards":[], "coords":[]}
obs = [0, 0]

for i in range(501):
    print("episode{0}----------".format(i))

    # act
    game.step("reset")
    new_observation, _, _, _ = game.step("move %s 0 %s" % (0, 0))
    coords = get_coords(get_extra(new_observation))
    obs = coords[[0,2]]
    print(obs)

    a_i = agent.act(obs)
    my_vec = agent_act(a_i)
    new_observation, reward, _, _ = game.step("move %s 0 %s" % (my_vec[0], my_vec[1]))
    evals['rewards'].append(reward)
    evals['coords'].append(coords)

    if reward > 0:
        print("reward: %s" % reward)
    if i%100 == 0:
        with open("pickle/eval_coords.pickle", mode="wb") as f:
            pickle.dump(evals, f)

print("finished!")