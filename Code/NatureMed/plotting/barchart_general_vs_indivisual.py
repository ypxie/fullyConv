import os, sys, pdb
import numpy as np
import matplotlib.pyplot as plt


DiseaseNames = ('G1', 'G2', 'G3', 'G4', 'G5')
DiseaseNum = len(DiseaseNames)

bar_width = 0.35                   # the width of the bars
ind = np.arange(DiseaseNum)        # the x locations for the groups
start_m = (1 - bar_width * 2) / 2
start_w = start_m +bar_width
fig, ax = plt.subplots()

# general performance
men_means = (20, 35, 30, 35, 27)
men_std = (2, 3, 4, 1, 2)
generic = ax.bar(ind+start_m, men_means, bar_width, color='#9cb9da', yerr=men_std)

# indivisual performance
women_means = (25, 32, 34, 20, 25)
women_std = (3, 5, 2, 3, 3)
indivisual = ax.bar(ind+start_w, women_means, bar_width, color='#f0c287', yerr=women_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('F1 score')
ax.set_title('Comparision between generic model and indivisual model')
ax.set_xticks(ind + 1.0 / 2)
ax.set_xticklabels(DiseaseNames)
ax.legend((generic[0], indivisual[0]), ('Generic', 'Indivisual'))
ax.grid(True)
fig.tight_layout()
plt.show()
