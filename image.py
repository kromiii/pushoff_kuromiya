import subprocess
import os
import gym
# import cv2
import time
import numpy as np
import json
import signal
from PIL import Image, ImageChops

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

for i in range(1000):
    # act
    new_observation, reward, end_episode, _ = game.step("reset")
    r,g,b = new_observation['image'][0].split()
    prev_r = np.asarray(r)
    for _ in range(100):
        
        a = [0, 0]

        a_i = np.random.randint(4)

        if a_i == 0:
            a[0] = 1
        elif a_i == 1:
            a[0] = -1
        elif a_i == 2:
            a[1] = 1
        elif a_i == 3:
            a[1] = -1

        for k in range(10):
            new_observation, reward, end_episode, _ = game.step("move %s 0 %s" % (a[0], a[1]))
            if k % 10 == 0:
                r,g,b = new_observation['image'][0].split()
                image = ImageChops.subtract(r,b)
                #image = image.convert("L")
                image = image.point(lambda x: 0 if x < 130 else 255)
                image.save('dice.png')
                image_r = np.asarray(image)
                diff_r = image_r - prev_r
                print(np.linalg.norm(diff_r))
                prev_r = image_r
            if reward > 0:
                print("reward: %s" % reward)

    # new_observation, reward, end_episode, _ = game.step("autograb")

    print("-" * 10)
