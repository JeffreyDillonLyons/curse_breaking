
'''Preamble'''

import numpy as np
import chaospy as ch
import pandas as pd
import time
import numpoly
import multiprocessing as mp
import chaospy as ch
import os

problem = { 
    'num_vars': 7,
    'names': ['alpha','beta','delta','gamma','e','f','h'],
    'bounds': [[0.44,0.68],
               [0.02,0.044],
               [0.71,1.15],
               [0.0226,0.0354],
               [0.03,0.055],
               [0.71,1.15],
               [0.02,0.03]]}

distributions = []

for idx,name in enumerate(problem['names']):
    a,b = problem['bounds'][idx]
    distributions.append(ch.Uniform(a,b))
    
joint = ch.J(*distributions)

def loader(check):
    
    path = './data/polynomials/'
    
    dick = numpoly.load(path+'master.npz')
    
    names = [name for name in pd.read_csv(path+'names.csv').names]
    
    dick = {names[i]:[i for i in dick.values()][i] for i in range(len(names))}
    
    if check:
        
        print('Polynomial loading successful...')
        print(f'Job list contains {len(animals)} polynomials')
        
    return names,dick

def sensitivity(i):
    names,dick = loader(check=False)
    ID = names[i]
    poly = dick[ID]
    start_time = time.perf_counter()
    print('Process', ID,'started...')
    
    s1 = ch.Sens_m(poly,joint)
    st = ch.Sens_t(poly,joint)
    
    df = pd.DataFrame(columns=['type','params','S1','ST','run_time'])
    df['type'] = 'owl'
    df.params = problem['names']
    df.run_time = time.perf_counter() - start_time
    df.S1 = s1
    df.ST = st
    
    path = f'./data/results/{ID}.csv'
    df.to_csv(path)
    print('.....',ID,'finished') 
    


if __name__ == '__main__':
    N = int(input('Please enter the number of processors to be used (dtype: int) >>> '))
    names,__ = loader(check=False)
    pool = mp.Pool(N)
    pool.map(sensitivity,[i for i in range(102)])
    pool.close()

