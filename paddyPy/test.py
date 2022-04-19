
'''Preamble'''

import numpy as np
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

#s = exec(open("temp.py").read())

def loader(N):
    
    path = './data/test/test.npz'
    
    animal = numpoly.load(path)['arr_0']
    
    animals = [(f'test_animal_{i}',animal) for i in range(N)]

    return animals



def test_function(a):
    N=a[1]
    i=a[0]
    animals = loader(N)
    animal = animals[i]
    ID = animal[0]
    print('Process', ID,'started...')
    poly = animal[1]
    start_time = time.perf_counter()
    
    s1 = ch.Sens_m(poly,joint)
    df = pd.DataFrame(columns=['type','params','S1','ST','run_time'],dtype='object')
    df['type'] = 'owl'
    df.params = problem['names']
    df.run_time = time.perf_counter() - start_time
    df.S1 = s1
    #df.ST = st
    
    path = f'./data/results/{ID}.csv'
    df.to_csv(path)
    print('.....',ID,'finished')    

if __name__ == '__main__':
    s = exec(open("./data/temp.py").read())
    N = int(input('Please enter number of processors for test (dtype:int) >>>'))
    #animals = loader(N)
    pool = mp.Pool(N)
    pool.map(test_function, [(i,N) for i in range(N)])
    pool.close()
