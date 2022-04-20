

'''Preamble'''

import numpy as np
import matplotlib.pyplot as plt
import chaospy as ch
import pandas as pd
import time
import numpoly
import multiprocessing as mp
import chaospy as ch

'''Parameters & settings'''
alpha = ch.Uniform(0.44, 0.68) 
beta = ch.Uniform(0.02, 0.044) #We take the same bounds as for the Sobol-Saltelli analysis
delta = ch.Uniform(0.71, 1.15)
gamma = ch.Uniform(0.0226, 0.0354)

joint = ch.J(alpha,beta,delta,gamma)

problem = {
    'num_vars': 4,
    'names': ['alpha','beta','delta','gamma'],
    'bounds': [[0.44,0.68],
               [0.02,0.044],
               [0.71,1.15],
               [0.0226,0.0354]]
} 

def loader():
    orders = [1]
    levels = [2,3,4]
    
    '''Owls'''
    global animals
    animals = []
    for i in orders:
        for j in levels:
            animals.append((f'gaussian_owl_e{i}_O{j}',numpoly.load(f'./data/gaussian/owl/poly_e{i}_O{j}.npz')['arr_0']))
            #animals.append((f'sparse_owl_e{i}_O{j}',numpoly.load(f'./data/sparse/owl/poly_e{i}_O{j}.npz')['arr_0']))
            #animals.append((f'gaussian_mouse_e{i}_O{j}',numpoly.load(f'./data/gaussian/mouse/poly_e{i}_O{j}.npz')['arr_0']))
           # animals.append((f'sparse_mouse_e{i}_O{j}',numpoly.load(f'./data/sparse/mouse/poly_e{i}_O{j}.npz')['arr_0']))
            
    print('Polynomial loading successful...')
    print(f'Job list contains {len(animals)} polynomials')
    return animals
    #print('Here is a selection >>>',animals[0],animals[2],animals[10])
            
            
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
    
    

problem = {
    'num_vars': 4,
    'names': ['alpha','beta','delta','gamma'],
    'bounds': [[0.44,0.68],
               [0.02,0.044],
               [0.71,1.15],
               [0.0226,0.0354]]
}   

def test_function(i):
    loader()
    animal = animals[i]
    ID = animal[0]
    print('Process', ID,'started...')
    poly = animal[1]
    start_time = time.perf_counter()
    
    s1 = ch.Sens_m(poly,joint)
    df = pd.DataFrame(columns=['type','params','S1','ST','run_time'])
    df['type'] = 'owl'
    df.params = problem['names']
    df.run_time = time.perf_counter() - start_time
    df.S1 = s1
    #df.ST = st
    
    path = f'./data/results/{ID}.csv'
    df.to_csv(path)
    print('.....',ID,'finished')
    

    
if __name__ == '__main__':
    pool = mp.Pool(2)
    pool.map(test_function, range(0,2))
    pool.close()
    
print('Complete')