import subprocess
import os
import gym
# import cv2
import time
import numpy as np
import json
import signal

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

n_dice = 4

def calc_reward(my_vec, coords, p_coords):
    rew = 0
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

for i in range(1000):
    # act
    new_observation, reward, end_episode, _ = game.step("reset")
    coords = get_coords(get_extra(new_observation))
    my_pos = np.array([0,0])
    for _ in range(500):
        p_coords = coords
        my_vec = np.array([0,0])

        a_i = np.random.randint(4)

        if a_i == 0:
            my_vec = [2,0]
        elif a_i == 1:
            my_vec = [-2,0]
        elif a_i == 2:
            my_vec = [0,2]
        elif a_i == 3:
            my_vec = [0,-2]

        my_pos += my_vec

        new_observation, reward, end_episode, _ = game.step("move %s 0 %s" % (my_vec[0], my_vec[1]))
        coords = get_coords(get_extra(new_observation))
        reward = calc_reward(my_vec,coords,p_coords)
        # print(train_obs)

        if reward > 0:
            print("reward: %s" % reward)    

    print("-" * 10)
