

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
import matplotlib as mpl
from SALib.sample import saltelli
from SALib.analyze import sobol
import seaborn as sns


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


x0 = 33                         # Initial conditions same as before
y0 = 6.2
X = [x0,y0]

'''
Input Paramters
-------------------
file path

'''

def plot_tile(old,title,order,path):

    fig = plt.figure(figsize=[12, 6])
    ax = fig.add_subplot(111)

    
    orderr = range(len(problem['names']))
    M = np.max(np.array(old))
    
    cmap = plt.get_cmap('Blues', M)
    
    img = ax.imshow(np.array(old).T, cmap=cmap, aspect='auto')
    t = np.arange(0,len(old))
    ax.set_xticks(t)
    
    
    
    norm = mpl.colors.Normalize(vmin=0, vmax=M-1)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm)
    
    p = np.linspace(0, M-1, M+1)
    tick_p = 0.5 * (p[1:] + p[0:-1])
    cb.set_ticks(tick_p)
    cb.set_ticklabels(np.arange(1,M+2))
    cb.set_label(r'quadrature order')
    ax.set_yticks(range(np.array(old).shape[1]))
    params = np.array(problem['names'])
    ax.set_yticklabels(params[orderr], fontsize=12)
    ax.set_xlabel('iteration')
    ax.set_title(title,fontsize='x-large')
    fig.set_size_inches((6,3))
    plt.savefig(f'{path}\\tile_plot_{order}.jpg',bbox_inches='tight',dpi=200)
    plt.tight_layout()
    plt.show()
    
def plot_hist(old,order,path):
    fig = plt.figure('adapt_hist', figsize=[4, 8])
    ax = fig.add_subplot(111, ylabel='max quadrature order',
                             title=f'Number of refinements = {old.shape[0]}')
                            
     # find max quad order for every parameter
    adapt_measure = np.max(old, axis=0)
    ax.bar(range(adapt_measure.size), height=adapt_measure)
    params = np.array(problem['names'])
    ax.set_xticks(range(0,adapt_measure.size,1))
    ax.set_yticks(range(0,adapt_measure.size,1))
    ax.set_xticklabels(params)
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.set_size_inches((6,3))
    plt.savefig(f'{path}\\hist_{order}.jpg',bbox_inches='tight',dpi=200)
    plt.show()   

def plot_poly(polynomials,sequence, path, order):

    from celluloid import Camera
 
    if sequence == 0:
        sequence = range(len(polynomials))
    else:
        sequence = sequence

    # polynomials = [polynomials[i] for i in sequence]
    
    t = np.linspace(0., 30, 1000)
    X = (33,6.2)

    def lotka(X, t, alpha, beta, delta, gamma):
        x, y = X
        dotx = x * (alpha - beta * y)
        doty = y * (-delta + gamma * x)
        return np.array([dotx, doty])

    samples = saltelli.sample(problem,256,calc_second_order=False)
    
    target = []
    for sample in samples:
        a,b,d,g = sample
        solution = integrate.odeint(lotka,X,t,args=(a,b,d,g)).T[0][910]
        target.append(solution)
    
    fig,ax = plt.subplots()
    ax.set_xlim(-10,100)
    camera = Camera(fig)
    
    targ = sns.kdeplot(target,color='red',label='Target distribution')
    camera.snap()
    
    evals = {}
    
    for seq in sequence:
        print(seq)
        evals[seq] = []
        for sample in samples:
            a,b,d,g = sample
            evals[seq].append(polynomials[seq](a,b,d,g))
   
    for key,seq in zip(evals.keys(),sequence):
        targ = sns.kdeplot(target,color='red')
        sns.kdeplot(evals[key], label = f'iteration: {seq}')
        camera.snap()

    # plt.show()
    animation = camera.animate(interval=1000)
    animation.save(f'animation_{order}.gif')
    
    
        
    
    
    
    
    
    
        
    
    
    
    


if __name__ == "__main__":

    root = r"C:\Users\jeffr\OneDrive\Documents\GitHub\new\adaptive_solver\data\Experimental_data_lotka_2\prey"
    
    root = os.path.abspath(root)
    
    order = 6
    
    df = pd.read_csv(root + f'\\run_file_{order}.csv')
    
    old = np.array([eval(i) for i in df.chosen_index][0:-1])
    
    title = f'Prey: truncation order = {order}'
    
    path = '.'
    
    '''Polynomials'''
    
    # polynomials = list(numpoly.load(root + f'\\poly_{order}.npz').values())
    
    
    # sequence = [0,1,4,5]
    
    # plot_poly(polynomials,sequence,path=root,order=order)
    
    # print(old)
    plot_tile(old,title,order,path=path)
    # plot_hist(old,order,path)
    
    
    
    
    
    