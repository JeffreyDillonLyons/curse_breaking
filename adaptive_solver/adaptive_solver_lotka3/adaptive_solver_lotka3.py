

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



##SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

# Chaospy
import chaospy as ch



def c(s):
    os.chdir(s)
    return os.getcwd()
    



'''
 Notes
-------------------------
TODO: 16:42 09/05/2022 > Check that addition of saving to dick in assign_errors doesn't blow things up.
''' 

""""""


x0 = 0.5
y0 = 1
z0 = 2

X = (x0,y0,z0)
t = np.linspace(0,30,1000)

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

def lotka(X,t,a,b,d,g,e,f,h):
    x, y, z = 0.5,1,2


    dotx = x * (a - b * y)
    doty = y * (-d + (g * x) - e * z)
    dotz = z * (-f + h * y)
    return np.array([dotx, doty, dotz])



distributions = [ ch.Uniform(a,b) for a,b in problem['bounds'] ]    


joint = ch.J(*distributions)

dick = {}

vectors = np.identity(len(joint), dtype="int")


gt = pd.read_csv('./data/redone_gt_lotka3_524288.csv',index_col=['type','params']).loc['owl','ST']
gt_norm = np.linalg.norm(gt)

growth=False
recurrence_algorithm='stieltjes'
rule='g'
tolerance=1e-10
scaling=3
n_max=50000



def _construct_lookup(
    orders, dists, growth, recurrence_algorithm, rules, tolerance, scaling, n_max,
):
    """
    Create abscissas and weights look-up table so values do not need to be
    re-calculatated on the fly.
    """
    x_lookup = []
    w_lookup = []

    for order, dist in zip(max_order_vector, dists):
        x_lookup.append([])
        w_lookup.append([])
        for orderr in range(max_order + 1):
            (abscissas,), weights = ch.generate_quadrature(
                order=orderr,
                dist=dist,
                growth=growth,
                recurrence_algorithm=recurrence_algorithm,
                rule=rule,
                tolerance=tolerance,
                scaling=scaling,
                n_max=n_max,
            )
            x_lookup[-1].append(abscissas)
            w_lookup[-1].append(weights)
    return x_lookup, w_lookup

def construct_wrapper(maxx):
    global max_order_vector, max_order
    global x_lookup, w_lookup

    max_order = maxx
    max_order_vector = max_order * np.ones(len(joint), dtype=int)
    # print(max_order_vector)

    x_lookup, w_lookup = _construct_lookup(
        orders=max_order_vector,
        dists=joint,
        growth=growth,
        recurrence_algorithm=recurrence_algorithm,
        rules=rule,
        tolerance=tolerance,
        scaling=scaling,
        n_max=5000,
    )

    return x_lookup, w_lookup


x_lookup, w_lookup = construct_wrapper(10)

def generate_candidates(index_set, P):
    global candidates, pre_candidates, back_neighbours

    pre_candidates = []
    candidates = []

    for j in range(0, len(joint)):
        pre_candidates.append(index_set + vectors[j])

    for candidate in pre_candidates:
        back_neighbours = []
        for j in range(0, len(joint)):
            back_neighbour = candidate - vectors[j]
            if np.all((back_neighbour > 1)):
                back_neighbours.append(tuple(back_neighbour))

        if np.all([neighbour in old for neighbour in back_neighbours]):
            candidates.append(tuple(candidate))

    # temp = []
    # for candidate in candidates:
        # # if candidate not in old:

         # if np.all(np.array(candidate) <= P) and np.linalg.norm(np.array(candidate),ord=1) <= (P+6):

            # temp.append(candidate)

    # candidates = temp
    
    for candidate in candidates:
        if candidate in old:
            candidates.remove(candidate)

    # temp = []
    # maxx = sum(np.max(np.array(old), axis=0))

    # for candidate in candidates:

        # if sum(np.max(np.array(old + [candidate]), axis=0)) > maxx:
            # temp.append(candidate)

    # candidates = temp

    return candidates


def sobol_error(vec):
    return np.linalg.norm(gt - vec) / gt_norm

def solver(old_set,target,expansion):



    global poly

    solver_time = time.perf_counter()

    nodes_list = []
    weights_list = []
    evals_list = []

    for index in old_set:
        nodes, weights = build_nodes_weights(index)
        weights = [weight * combinator(index,vectors) for weight in weights]

        nodes_list += nodes
        weights_list += weights

    for node in nodes_list:

        if node in dick.keys():
            evals_list.append(dick[node])

        else:
            a, b, d, g, e, f, h = node
            solution = integrate.odeint(lotka, X, t, args=(a, b, d, g, e, f, h)).T[target][910]
            evals_list.append(solution)
            dick[node] = solution

    nodes_list = np.array(nodes_list).T

    polly, uhat = ch.fit_regression(expansion, nodes_list, evals_list, retall = 1)
    poly.append(polly)

    # print('Solver_time >>>', time.perf_counter() - solver_time)
    # print('Weight sum >>>', sum(weights_list))
    return len(weights_list), uhat



def assign_errors(active_set,target):
    global active_errors, active, candidates, current_errors, new
    
    active_errors = []

    
    if np.any(active_set in old):
        print('oops')
    
    # maxx = sum(np.max(np.array(old), axis=0))
    
    # for multi_index in active_set:
        # if (step > 0) and sum(np.max(np.array(old + [multi_index]), axis=0)) <= maxx:
            # active_set.remove(multi_index)
        


    for multi_index in active_set:
        nodes, _ = build_nodes_weights(multi_index)
        current_errors = []

        for node in nodes:
            a, b, d, g, e, f, h = node
            if np.isnan(poly[-1](a, b, d, g, e, f, h)):
                poly_eval = 0
            else:
                poly_eval = poly[-1](a, b, d, g, e, f, h)

            if node in dick.keys():
                # a, b, d, g = node
                current_errors.append(abs(dick[node] - poly_eval))

            else:

                solution = integrate.odeint(lotka, X, t, args=(a, b, d, g ,e ,f ,h )).T[target][910]

                current_errors.append(abs(solution - poly_eval))
                dick[node] = solution

        active_errors.append(np.mean(current_errors))

    active = sorted(list(zip(active_set, active_errors)), key=lambda x: x[1])
    for i in active:
        if step > 1:
            if i[0] in old:
                active.remove(i)

    return active

  
def algorithm(P, species, TOL, merge):
    
    global dick, old, candidates, poly, active, global_errors, no_nodes,step
    
    '''Initialise'''
    
    if species == 'mouse':
        target = 0
        
    elif species == 'snake':
        target = 1
      
    elif species == 'owl':
        target = -1
    
    # seed = (2,2,1,2)
    expansion = ch.generate_expansion(P, joint, normed = True)
    exponents = ch.lead_exponent(expansion, graded=True)
    vectors = np.identity(len(joint), dtype='int')

    date_today = datetime.date.today()
    start_time = time.perf_counter()

    step = 0

    
    old = [tuple(np.zeros(len(joint),dtype='int'))]
    active = []
    poly = []
    uhats = []
    
    local_errors = []
    global_errors = []
    means = []
    
    names = ['alpha','beta','delta','gamma','e','f','h']
    
    df = pd.DataFrame(columns=['chosen_index','local_error','global_error','no_nodes','run_time'])
    df_indices = pd.DataFrame(columns=names,dtype=object)
    df_indices_s1 = pd.DataFrame(columns=names,dtype=object)

    
    '''Execute zeroth step'''
    
    # trivial = [seed]
    number_nodes,uhat = solver(old,target,expansion)
    uhats.append(uhat)
    assign_errors(old,target)
    
    st = sense_t(uhat,exponents,expansion)
    s1 = sense_main(uhat,exponents,expansion)
    means.append(uhat[0])
    

    global_errors.append(sobol_error(st))

    print("Global error >>>", global_errors[-1])
    print("Step time >>>", time.perf_counter() - start_time, "seconds")
    print("-" * 10, "break", "-" * 10)

  
    '''Main loop'''
    
    while (global_errors[-1] > TOL or np.isnan(global_errors[-1])) and len(active)>0:
        
        start_time = time.perf_counter()
        
        # print('Active >>>',active)

        chosen_index = active[-1][0]
        local_errors.append(active[-1][1])
        active.pop()

        old.append(chosen_index)

        
        print('Chosen index >>>', chosen_index)
        
        number_nodes,uhat = solver(old,target,expansion)
        uhats.append(uhat)
        
        candidates = generate_candidates(chosen_index,P)
        stripped_active = [i[0] for i in active] + [j for j in candidates]
        active = assign_errors(stripped_active,target)
        
        sobol_time = time.perf_counter() 

        st = sense_t(uhat,exponents,expansion)
        s1 = sense_main(uhat,exponents,expansion)
        means.append(uhat[0])
        
        # print('Sobol time >>>', time.perf_counter() - sobol_time)
        

        global_errors.append(sobol_error(st))

        print("Global error >>>", global_errors[-1])

        """Save data"""
        run_time = time.perf_counter() - start_time

        numpoly.savez(f'../data/lotka3/{species}/poly/poly_{P}+{date_today}.npz',*poly)
        np.savez(f'../data/lotka3/{species}/poly/uhat_{P}+{date_today}.npz',*uhats)
        
        df_indices = df_indices.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3], 'e': st[4], 'f':st[5],'h':st[6]}, ignore_index=True)
        df_indices_s1 = df_indices_s1.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3], 'e': st[4], 'f':st[5],'h':st[6]}, ignore_index=True)
        df = df.append({'chosen_index': chosen_index,'local_error':local_errors[-1],                               'global_error':global_errors[-1],'no_nodes':number_nodes, 'run_time':run_time}, ignore_index=True)
        
        df.to_csv(f'../data/lotka3/{species}/run_file_{P}+{date_today}.csv')
        df_indices.to_csv(f'../data/lotka3/{species}/total_order_indices_{P}+{date_today}.csv')
        df_indices_s1.to_csv(f'../data/lotka3/{species}/first_order_indices_{P}+{date_today}.csv')

        print('Step time >>>', time.perf_counter() - start_time, 'seconds')
        print('-'*10,'break','-'*10)
        
        step += 1
        
    print('Congratulations, the algorithm has converged!')
    print('Here are the results...')
    print('-'*20)
    print(f'ST_alpha:{st[0].round(4)}, ST_beta:{st[1].round(4)}, ST_delta:{st[2].round(4)},\
            ST_gamma:{st[3].round(4)}, ST_e:{st[4].round(4)}, ST_f:{st[5].round(4)}, ST_h:{st[6].round(4)}')                                    
              
    print(f'GT_alpha:{gt[0].round(4)}, GT_beta:{gt[1].round(4)}, GT_delta:{gt[2].round(4)}, GT_gamma:{gt[3].round(4)}, GT_e:{gt[4].round(4)}, GT_f:{gt[5].round(4)}, GT_h:{gt[6].round(4)} ')
    print(f'The final grid contains {number_nodes} nodes.')
    print(f'The total run time was {df.run_time.sum()}seconds, not bad!')
    
    if merge:
    
        merged_set = merge_sets(old,active)
        
        number_nodes,uhat = solver(merged_set, target, expansion)
            
        st = sense_t(uhat,exponents,expansion)
        s1 = sense_main(uhat,exponents,expansion)
        
        print('-'*10,'MERGED','-'*10)
        print(f'ST_alpha:{st[0].round(4)}, ST_beta:{st[1].round(4)}, ST_delta:{st[2].round(4)},\
            ST_gamma:{st[3].round(4)}, ST_e:{st[4].round(4)}, ST_f:{st[5].round(4)}, ST_h:{st[6].round(4)}')                                    
              
        print(f'GT_alpha:{gt[0].round(4)}, GT_beta:{gt[1].round(4)}, GT_delta:{gt[2].round(4)}, GT_gamma:{gt[3].round(4)}, GT_e:{gt[4].round(4)}, GT_f:{gt[5].round(4)}, GT_h:{gt[6].round(4)} ')
        
        global_errors.append(sobol_error(st))


        print('Global error >>>', global_errors[-1])
            
        '''Save data'''
        run_time = time.perf_counter() - start_time
     
        numpoly.savez(f'../data/lotka3/{species}/poly/poly_{P}+{date_today}.npz',*poly)
        
        np.savez(f'../data/lotka3/{species}/poly/uhat_{P}+{date_today}.npz',*uhats)
        
        df_indices = df_indices.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3], 'e': st[4], 'f':st[5],'h':st[6]}, ignore_index=True)
        df_indices_s1 = df_indices_s1.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3], 'e': st[4], 'f':st[5],'h':st[6]}, ignore_index=True)
        df = df.append({'chosen_index': chosen_index,'local_error':local_errors[-1],                               'global_error':global_errors[-1],'no_nodes':number_nodes, 'run_time':run_time}, ignore_index=True)
        
        df.to_csv(f'../data/lotka3/{species}/run_file_{P}+{date_today}.csv')
        df_indices.to_csv(f'../data/lotka3/{species}/total_order_indices_{P}+{date_today}.csv')
        df_indices_s1.to_csv(f'../data/lotka3/{species}/first_order_indices_{P}+{date_today}.csv')
        
def merge_sets(old_set,active):

    stripped_active = [i[0] for i in active]
    merged = old + stripped_active
    
    return merged 

def combinator(current_index,vectors):
    coeff = 1

    for vector in vectors:

        if tuple(np.array(current_index, dtype="int") + vector) in old:

            coeff += -1

    return coeff


def build_nodes_weights(current_index):

    nodestack = []
    weightstack = []

    """Nodes"""

    for index, element in enumerate(current_index):
        nodestack.append([])
        nodestack[index] = list(x_lookup[index][element])

        
    nodes = product(*nodestack)
        
    '''Weights'''
    
    for index,element in enumerate(current_index):
        weightstack.append([])
        weightstack[index] = list(w_lookup[index][element])

    weights = product(*weightstack)

    return nodes,weights
        
def sense_main(uhat,exponents,expansion):

    dim = len(joint)
    s1 = np.zeros(dim)
    
    variance = np.sum(np.array(uhat[1:])**2)
    
    for variable,name in enumerate(expansion.names):
        mask = np.ones(dim)
        mask[variable] = False
        
        for idx,exponent in enumerate(exponents):
            if exponent[variable] > 0 and np.all(exponent*mask == 0):
                s1[variable] += uhat[idx]**2
                
    s1 = s1 / variance
    
    return s1        

def sense_t(uhat,exponents,expansion):

    dim = len(joint)
    st = np.zeros(dim)
    
    variance = np.sum(np.array(uhat[1:])**2)
    
    for variable,name in enumerate(expansion.names):


        mask = np.ones(dim)
        mask[variable] = False
        
        for idx,exponent in enumerate(exponents):
            if exponent[variable] > 0 and np.all(exponent*mask == 0):
                st[variable] += uhat[idx]**2
                
            if exponent[variable] > 0 and np.any(exponent*mask != 0):
                st[variable] += uhat[idx]**2
                
    st = st / variance
    
    return st

algorithm(3,'owl',0.2, merge=True)
