

'''Preamble'''

import numpy as np
import chaospy as ch
from scipy import integrate
import pandas as pd
import time
import json
import pandas as pd
import datetime
import numpoly
from itertools import product
from collections import OrderedDict
import re
import os
import matplotlib.pyplot as plt


def c(s):
    os.chdir(s)
    return os.getcwd()
    
problem = {
    'num_vars': 4,
    'names': ['alpha','beta','delta','gamma'],
    'bounds': [[0.44,0.68],
               [0.02,0.044],
               [0.71,1.15],
               [0.0226,0.0354]]
}
number = 4
old = pd.read_csv(f'../data/Experimental_data_lotka_2/run_file_{number}+2022-05-9.csv').old
old = [eval(i) for i in old]

	
def tile_plot(old):

    fig = plt.figure(figsize=[12, 6])
    ax = fig.add_subplot(111)

    
    order = range(len(problem['names']))
    M = np.max(np.array(old))
    cmap = plt.get_cmap('Purples', M)
    plt.imshow(np.array(old).T, cmap=cmap, aspect='auto')
    norm = mpl.colors.Normalize(vmin=0, vmax=M - 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm)
    p = np.linspace(0, M - 1, M + 1)
    tick_p = 0.5 * (p[1:] + p[0:-1])
    cb.set_ticks(tick_p)
    cb.set_ticklabels(np.arange(M))
    cb.set_label(r'quadrature order')
    ax.set_yticks(range(np.array(old).shape[1]))
    params = np.array(problem['names'])
    ax.set_yticklabels(params[order], fontsize=12)
    ax.set_xlabel('iteration')
    plt.tight_layout()
    plt.show()
print(old)