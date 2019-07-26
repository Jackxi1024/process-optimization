#!/usr/bin/env python3

import sys
import time
import numpy as np
import pandas as pd
import os
import math
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

FILE = None

if FILE is None:
    root = tk.Tk()
    root.withdraw()
    FILE = filedialog.askopenfilename()

DIR = os.path.dirname(FILE)
df = pd.read_csv(FILE)

fitnesses = df[["capex", "roi"]]

comparison_y = [5.826, 5.672, 5.484, 5.276, 5.033, 4.758, 4.385]
comparison_x = [0.723, 0.715, 0.681, 0.611, 0.479, 0.254, -0.205]


plt.figure(figsize=(20/3,15/3))
plt.plot(fitnesses["roi"], fitnesses["capex"], '.k')
plt.plot(comparison_x, comparison_y, 'xr')
plt.ylabel("F1(p)")
plt.xlabel("F2(p)")
plt.axis([-1, 1, 4, 8])

plt.savefig(DIR+"/dual_fitness.eps")



def animate(i):
    characteristic_iterations = 1500
    preoptimization_iterations = 300

    iters = max(i-preoptimization_iterations, 0)
    w1 = 0.5+0.5*math.cos(math.pi*iters/characteristic_iterations)
    w2 = 1 - w1
    plt.plot(fitnesses.loc[0:i+1, "roi"], fitnesses.loc[0:i+1, "capex"], color='gray', marker='.', linestyle='')
    plt.plot(fitnesses.loc[i+1, "roi"], fitnesses.loc[i+1, "capex"], color='black', marker='.', linestyle='')
    plt.plot(comparison_x, comparison_y, 'xr')
    plt.text(0.5, 7.8, "w1 = %1.3f" % w1)
    plt.text(0.5, 7.6, "w2 = %1.3f" % w2)


# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, bitrate=8e6)

# mpl.rcParams['figure.dpi'] = 500
# fig = plt.figure(figsize=(19.2,10.8))
# plt.ylabel("F1(p)")
# plt.xlabel("F2(p)")
# plt.axis([-1, 1, 4, 8])
# ani = animation.FuncAnimation(fig, animate, frames=60, repeat=True)
# ani.save(DIR + "/dual_fitness.mp4", writer=writer)

# len(fitnesses["roi"])-5












