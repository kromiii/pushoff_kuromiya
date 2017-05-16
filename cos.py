import subprocess
import os
# import cv2
import time
import numpy as np
import json
import signal

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym


linux = True

game_processes = []

# port render_freq msg_freq server
if linux:
    game_processes.append(
        subprocess.Popen("./game_linux.x86_64 5000 10 10 1 aaaaaaaaaa", shell=True, stdout=subprocess.PIPE,
                         preexec_fn=os.setsid))
else:
    game_processes.append(
        subprocess.Popen("open -a game_mac.app --args 5000 10 10 1 aaaaaaaaaa", shell=True, stdout=subprocess.PIPE,
                         preexec_fn=os.setsid))

time.sleep(7)

game = gym.make('Unity-v0')
game.configure("5000")

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
    ret = coords[coords != 0.0]
    return ret

def omit_height(coords):
    return coords[[0,2,3,5,6,8,9,11]]

def random_action_func():
    return np.random.randint(4)

n_dice = 4
obs_size = n_dice*2 + 2
n_actions = 4

q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(obs_size, n_actions,n_hidden_layers=2, n_hidden_channels=50)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
gamma = 0.95
explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=random_action_func)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
agent = chainerrl.agents.DoubleDQN(q_func, optimizer, replay_buffer, gamma, explorer,replay_start_size=500, update_frequency=1,target_update_frequency=100, phi=phi)
agent.load('agent')

def calc_reward(my_vec, coords, p_coords, t):
    rew = 0
    if t < 30:
        return 0
    for i in range(n_dice):
        if (coords[(3*i+1)] < - 100):
            continue
        r = 0
        d_vec = coords[[3*i,3*i+2]] - p_coords[[3*i,3*i+2]]
        d_norm = np.linalg.norm(d_vec)
        if d_norm < 0.1:
            continue
        d_vec = d_vec / d_norm
        r = np.dot(my_vec, d_vec)
        rew += r
    return rew

for i in range(5000):
    # act
    new_observation, reward, end_episode, _ = game.step("reset")
    coords = get_coords(get_extra(new_observation))
    my_pos = np.array([0,0])
    my_vec = np.array([0,0])
    train_obs = np.append(my_vec, omit_height(coords))
    for t in range(300):
        p_coords = coords

        a_i = agent.act_and_train(train_obs, reward)

        if a_i == 0:
            my_vec = [1,0]
        elif a_i == 1:
            my_vec = [-1,0]
        elif a_i == 2:
            my_vec = [0,1]
        elif a_i == 3:
            my_vec = [0,-1]

        my_pos += my_vec

        new_observation, reward, end_episode, _ = game.step("move %s 0 %s" % (my_vec[0], my_vec[1]))
        coords = get_coords(get_extra(new_observation))
        reward = calc_reward(my_vec,coords,p_coords, t)
        train_obs = np.append(my_vec, omit_height(coords))
        #print(train_obs)

        if reward > 0:
            print("reward: %s" % reward)    

    print("-" * 10)
    print(i)
    if i%100 == 0:
        agent.save('agent')
