#! /usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.style.use('ggplot')

with open('eval_image.pickle', 'rb') as f:
    eval_image = pd.DataFrame(pickle.load(f))
with open('eval_coords.pickle', 'rb') as f:
    eval_coords = pd.DataFrame(pickle.load(f))
with open('eval_cos.pickle', 'rb') as f:
    eval_cos = pd.DataFrame(pickle.load(f))
image = eval_image.rewards.sum()
coords = eval_coords.rewards.sum()
cos = eval_cos.rewards.sum()

data = {"image":[image], "coords":[coords], "cos":[cos]}
df = pd.DataFrame(data)

ax = df.plot(kind="bar", title="total reward in 500 steps")
ax.set(ylabel='point')
plt.show()