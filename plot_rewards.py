#! /usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.style.use('ggplot')

with open('rewards_image.pickle', 'rb') as f:
    rewards_image = pickle.load(f)
with open('rewards_coords.pickle', 'rb') as f:
    rewards_coords = pickle.load(f)
with open('rewards_cos.pickle', 'rb') as f:
    rewards_cos = pickle.load(f)
mean_rew_image = np.mean(np.array(rewards_image[:5500]).reshape(-1, 100), axis=1)
mean_rew_coords = np.mean(np.array(rewards_coords[:5500]).reshape(-1, 100), axis=1)
mean_rew_cos = np.mean(np.array(rewards_cos[:5500]).reshape(-1, 100), axis=1)

data = {"image":mean_rew_image, "coords":mean_rew_coords, "cos":mean_rew_cos}
df = pd.DataFrame(data)

ax = df.plot(title="mean reward by every 100 step")
ax.set(xlabel='step x100', ylabel='mean rewards / 100 times')
plt.show()