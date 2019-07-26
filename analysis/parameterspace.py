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

iterations = df["iter"]
fitnesses = df["fitness_function"]
parameters = df[["eductF_n", "fracA_n", "eductT_n", "heatexT_n", "purgeR_n"]]

i = 1
# plt.figure(); 
plt.figure(figsize=(20/3,15/3))

for parameter in parameters:
    plt.subplot(2, 3, i)
    i += 1
    
    plt.plot(parameters[parameter], fitnesses, '.k')
    plt.axhline(y=max(fitnesses), linewidth=1, color='r')
    plt.ylabel("F(p)")
    plt.xlabel(parameter)
    plt.subplots_adjust(hspace = 0.4, wspace = 0.55, top = 0.95, bottom = 0.1, left = 0.1, right = 0.95)
    plt.axis([0, 1, 0, 1])


plt.subplot(2,3,6)
plt.plot(iterations, fitnesses, '.k')
plt.axhline(y=max(fitnesses), linewidth=1, color='r')
plt.ylabel("F(p)")
plt.xlabel("#evaluation")
plt.subplots_adjust(hspace = 0.4, wspace = 0.55, top = 0.95, bottom = 0.1, left = 0.1, right = 0.95)
plt.axis([0, None, -5, 1])

plt.savefig(DIR+"/parameterspace.eps")


















