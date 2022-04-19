
'''Preamble'''

import numpy as np
import matplotlib.pyplot as plt
import chaospy as ch
import pandas as pd
import time
import numpoly
import multiprocessing as mp
import chaospy as ch

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

def loader(check=True):
    orders = [1]
    levels = [2,3,4]
    
    global animals
    animals = []
    for i in orders:
        for j in levels:
            animals.append((f'gaussian_owl_e{i}_O{j}',numpoly.load(f'./data/gaussian/owl/poly_e{i}_O{j}.npz')['arr_0']))
            animals.append((f'sparse_owl_e{i}_O{j}',numpoly.load(f'./data/sparse/owl/poly_e{i}_O{j}.npz')['arr_0']))
            animals.append((f'gaussian_mouse_e{i}_O{j}',numpoly.load(f'./data/gaussian/mouse/poly_e{i}_O{j}.npz')['arr_0']))
            animals.append((f'sparse_mouse_e{i}_O{j}',numpoly.load(f'./data/sparse/mouse/poly_e{i}_O{j}.npz')['arr_0']))
            
    if check:
        
        print('Polynomial loading successful...')
        print(f'Job list contains {len(animals)} polynomials')
    
    return animals
    
loader(check=True)

def sensitivity(animal):
    ID = animal[0]
    poly = animal[1]
    start_time = time.perf_counter()
    
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

if __name__ == '__main__':
    animals = loader(check=False)
    pool = mp.Pool(2)
    pool.map(sensitivity, range(len(animals))
    pool.close()

