

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


def c(s):
    os.chdir(s)
    return os.getcwd()
''''''




x0 = 33                         # Initial conditions same as before
y0 = 6.2
X = [x0,y0]
t = np.linspace(0., 30, 1000)

#Model

def lotka(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

#Parameter space

problem = {
    'num_vars': 4,
    'names': ['alpha','beta','delta','gamma'],
    'bounds': [[0.44,0.68],
               [0.02,0.044],
               [0.71,1.15],
               [0.0226,0.0354]]
}
##Parameters
alpha = ch.Uniform(0.44, 0.68) #We choose uniform distributions to reflect our lack of knowledge about the relative likelihood functions
beta = ch.Uniform(0.02, 0.044) #We take the same bounds as for the Sobol-Saltelli analysis
delta = ch.Uniform(0.71, 1.15)
gamma = ch.Uniform(0.0226, 0.0354)

joint = ch.J(alpha,beta,delta,gamma) #The input paramter distributions are assumed to be independent so we may easily construct the joint input probability distribution.



gt_prey = pd.read_csv('../data/indices_328000_GT.csv').ST
gt_norm = np.linalg.norm(gt_prey)


dick = {}

vectors = np.identity(len(joint), dtype='int')

growth=False
recurrence_algorithm='stieltjes'
rule='g'
tolerance=1e-10
scaling=3
n_max=50000


def _construct_lookup(
        orders,
        dists,
        growth,
        recurrence_algorithm,
        rules,
        tolerance,
        scaling,
        n_max,
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
        n_max=5000)

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

    temp = []
    for candidate in candidates:
        # if candidate not in old:
        if (np.all(np.array(candidate) <= P) and np.linalg.norm(np.array(candidate),ord=1) <= (P+3)):
            temp.append(candidate)

    candidates = temp

#     temp = []
#     maxx = sum(np.max(np.array(old), axis=0))

#     for candidate in candidates:

#         if sum(np.max(np.array(old + [candidate]), axis=0)) > maxx:
#             temp.append(candidate)

#     candidates = temp

    return candidates

def sobol_error(vec):
    return np.linalg.norm(gt_prey - vec) / gt_norm


def solver(old_set,target):

    global poly

    solver_time = time.perf_counter()

    nodes_list = []
    weights_list = []
    evals_list = []

    for index in old_set:
        nodes, weights = build_nodes_weights(index)
        weights = [weight * combinator(index) for weight in weights]

        nodes_list += nodes
        weights_list += weights

    for node in nodes_list:

        if node in dick.keys():
            evals_list.append(dick[node])

        else:
            a, b, d, g = node
            solution = integrate.odeint(lotka, X, t, args=(a, b, d, g)).T[target][910]
            evals_list.append(solution)
            dick[node] = solution

    nodes_list = np.array(nodes_list).T

    polly, uhat = ch.fit_regression(expansion,nodes_list, evals_list, retall = 1)
    poly.append(polly)
    print('Solver_time >>>', time.perf_counter() - solver_time)
    print('Weight sum >>>', sum(weights_list))
#     print(len(uhat))
    return len(weights_list), uhat


def assign_errors(active_set):
    global active_errors, active, candidates, current_errors, new
    active_errors = []
    
#     maxx = sum(np.max(np.array(old), axis=0))

#     for multi_index in active_set:
#         if (step > 0) and sum(np.max(np.array(old + [multi_index]), axis=0)) <= maxx:
#             active_set.remove(multi_index)
        

    for multi_index in active_set:
        nodes, _ = build_nodes_weights(multi_index)
        current_errors = []

        for node in nodes:
            a, b, d, g = node
            if np.isnan(poly[-1](a, b, d, g)):
                poly_eval = 0
            else:
                poly_eval = poly[-1](a, b, d, g)

            if node in dick.keys():
                a, b, d, g = node
                current_errors.append(abs(dick[node] - poly_eval))

            else:
                solution = integrate.odeint(lotka, (33, 6.2), t, args=(a, b, d, g)).T[0][910]
                current_errors.append(abs(solution - poly_eval))

        active_errors.append(np.mean(current_errors))

    active = sorted(list(zip(active_set, active_errors)), key=lambda x: x[1])

    return active
    # active = [i for i in OrderedDict((tuple(x[0]), x) for x in active).values()]if np.isnan(poly[-1](1,2,3,4)):

def algorithm(P,species):
    
    global dick, old, candidates, poly, active, global_errors, no_nodes,step,expansion
    
    '''Initialise'''
    
    if species == 'prey':
        target = 0
        
    elif species == 'predator':
        target = 1
    
    seed = (2,2,1,2)
    expansion = ch.generate_expansion(P, joint, normed = True)
    exponents = ch.lead_exponent(expansion, graded=True)
    vectors = np.identity(len(joint), dtype='int')
    date_today = datetime.date.today()
    start_time = time.perf_counter()
    
    step = 0
    
    old = [(1,1,1,1)]
    active = []
    poly = []
    
    local_errors = []
    global_errors = []
    
    names = ['alpha','beta','delta','gamma']
    
    df = pd.DataFrame(columns=['chosen_index','local_error','global_error','no_nodes','run_time'],dtype=object)
    df_indices = pd.DataFrame(columns=['alpha','beta','delta','gamma'],dtype=object)
    df_indices_s1 = pd.DataFrame(columns=['alpha','beta','delta','gamma'],dtype=object)

    
    '''Execute zeroth step'''
    
    trivial = [seed]
    number_nodes,uhat = solver(old,target)
    assign_errors(old)
    
    
    st = sense_t(uhat,exponents)
    s1 = sense_main(uhat,exponents)
    
    global_errors.append(sobol_error(st))
    
    print('Global error >>>', global_errors[-1])
    print('Step time >>>', time.perf_counter() - start_time, 'seconds')
    print('-'*10,'break','-'*10)


    '''Main loop'''
    
    while (global_errors[-1] > 0.2 or np.isnan(global_errors[-1])) and len(active)>0:
        
        start_time = time.perf_counter()
        
#         print('Active >>>',active)
        chosen_index = active[-1][0]
        local_errors.append(active[-1][1])
        active.pop()
    
        old.append(chosen_index)
        
        print('Chosen index >>>', chosen_index)
        
        number_nodes,uhat = solver(old,target)
        
        candidates = generate_candidates(chosen_index,P)
        stripped_active = [i[0] for i in active] + [j for j in candidates]
        active = assign_errors(stripped_active)
        
        sobol_time = time.perf_counter() 

        st = sense_t(uhat,exponents)
        s1 = sense_main(uhat,exponents)
        
        print('Sobol time >>>', time.perf_counter() - sobol_time)
        
        global_errors.append(sobol_error(st))

        print('Global error >>>', global_errors[-1])
        
        '''Save data'''
        run_time = time.perf_counter() - start_time
        
        numpoly.savez(f'../data/lotka2/{species}/poly_{P}+{date_today}.npz',*poly)
        
        df_indices = df_indices.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3]          }, ignore_index=True)
        df_indices_s1 = df_indices_s1.append({'alpha': s1[0], 'beta': s1[1], 'delta': s1[2], 'gamma': s1[3]          }, ignore_index=True)
        df = df.append({'chosen_index': chosen_index,'local_error':local_errors[-1],                               'global_error':global_errors[-1],'no_nodes':number_nodes, 'run_time':run_time}, ignore_index=True)
        
        df.to_csv(f'../data/lotka2/{species}/run_file_{P}+{date_today}.csv')
        df_indices.to_csv(f'../data/lotka2/{species}/total_order_indices_{P}+{date_today}.csv')
        df_indices_s1.to_csv(f'../data/lotka2/{species}/first_order_indices_{P}+{date_today}.csv')

        print('Step time >>>', time.perf_counter() - start_time, 'seconds')
        print('-'*10,'break','-'*10)
        
        step += 1
        
    print('Congratulations, the algorithm has converged!')
    print('Here are the results...')
    print('-'*20)
    print(f'ST_alpha:{st[0].round(10)}, ST_beta:{st[1].round(10)}, ST_delta:{st[2].round(10)}, ST_gamma:{st[3].round(10)}')
    print(f'GT_alpha:{gt_prey[0].round(10)}, GT_beta:{gt_prey[1].round(10)}, GT_delta:{gt_prey[2].round(10)}, GT_gamma:{gt_prey[3].round(10)}')
    print(f'The final grid contains {number_nodes} nodes.')
    print(f'The total run time was {df.run_time.sum()}seconds, not bad!')
          

def combinator(current_index):
    
    coeff = 1
    
    for vector in vectors:
        
        if tuple(np.array(current_index, dtype='int') + vector) in old:
            
            coeff += -1
            
    return coeff 

def build_nodes_weights(current_index):
    
    nodestack = []
    weightstack = []
    
    '''Nodes'''
    
    for index,element in enumerate(current_index):
        nodestack.append([])
        nodestack[index] = list(x_lookup[index][element])
        
    nodes = nodestack[0]
    
    for i in range(1,len(nodestack)):
        nodes = product(nodes,nodestack[i])
        
    nodes = [(a,b,c,d) for (((a,b),c),d) in nodes]
    
    '''Weights'''
    
    for index,element in enumerate(current_index):
        weightstack.append([])
        weightstack[index] = list(w_lookup[index][element])
        
    weights = weightstack[0]
    
    for i in range(1,len(weightstack)):
        weights = product(weights,weightstack[i])
        
    weights = [(a*b*c*d) for (((a,b),c),d) in weights]
    
    
    return nodes,weights

def sense_main(uhat,exponents):

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

def sense_t(uhat,exponents):
    
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
    
    
        
algorithm(3,'prey')









