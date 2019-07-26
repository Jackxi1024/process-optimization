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


















