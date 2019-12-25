import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(8, 8))
labelled_data = zip(y.transpose(), ('A', 'F', 'G'), ('b', 'g', 'r'))
fig.suptitle('Three Trends', fontsize=16)

for i, ld in enumerate(labelled_data):
    ax = axes[i]
    ax.plot(x, ld[0], label=ld[1], color=ld[2])
    ax.set_ylabel('Sum')
    ax.legend(loc='upper left', framealpha=0.5, prop={'size': 'small'})
axes[-1].set_xlabel('Date')
plt.show()