

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

from ema_workbench.connectors.vensim import (
    VensimModel,
    load_model,
    set_value,
    run_simulation,
    get_data,
    be_quiet,
    vensimDLLwrapper,
    VensimModel
)
from ema_workbench import (
    ema_logging,
    MultiprocessingEvaluator,
    RealParameter,
    ScalarOutcome,
    Scenario,
    Samplers,
    save_results
)



# TODO > model loading into algorithm
# TODO > polynomial args arbitray length







def get_gt(dimension):
    
    gt_path = fr'C:\Users\jeffr\OneDrive\Documents\Education\Thesis\databank\ETM\sobol_{dimension}_2048.csv'
    gt = pd.read_csv(gt_path).ST
    gt_norm = np.linalg.norm(gt)
    
    return gt,gt_norm

def get_evals(dimension):

    if dimension == 5:
        dick_path = r"C:\Users\jeffr\OneDrive\Documents\Education\Thesis\databank\ETM\runs_dict_mo10_dim5_sparse.json"
        
    elif dimension == 8:
        dick_path = r"C:\Users\jeffr\OneDrive\Documents\Education\Thesis\databank\ETM\runs_dict_mo8_dim8_sparse.json"
        
    elif dimension == 10:
        dick_path = r"C:\Users\jeffr\OneDrive\Documents\Education\Thesis\databank\ETM\runs_dict_mo7_dim10_sparse.json"
   
    with open(dick_path,'rb') as f:
        dick = json.loads(f.read())
    print('dick_loaded')
    
    return dick
    
def get_joint(dimension, parameter_names):

    current_names = parameter_names[0:dimension]
    
    with open("../data/variable_settings.json") as f:
        var_settings = json.loads(f.read())

    bounds = [var_settings[name] for name in current_names]
    
    distributions = [ch.Uniform(bound[0], bound[1]) for bound in bounds]

    joint_distribution = ch.J(*distributions)
    
    return joint_distribution,bounds
        
def return_last(x):
    return x[-1]


def setup_vensimmodel(model_file, working_directory, parameter_names, bounds):
    """

    Parameters
    ----------
    model_file
    working_directory
    parameter_names
    bounds

    Returns
    -------

    """
    model = VensimModel("vensimmodel", wd=working_directory, model_file=model_file)

    uncertainties = []
    for name, bound in zip(parameter_names, bounds):
        lower, upper = bound
        uncertainty = RealParameter(name, lower, upper)
        uncertainties.append(uncertainty)
    model.uncertainties = uncertainties

    model.outcomes = [ScalarOutcome("fraction renewables", function=return_last)]

    return model            



def _construct_lookup(orders, dists):
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
                growth=False,
                recurrence_algorithm='stieltjes',
                rule='g',
                tolerance=1e-10,
                scaling=3,
                n_max=50000,
            )
            x_lookup[-1].append(abscissas)
            w_lookup[-1].append(weights)
    return x_lookup, w_lookup

def construct_wrapper(maxx,joint_distribution):
    global max_order_vector, max_order
    global x_lookup, w_lookup

    max_order = maxx
    max_order_vector = max_order * np.ones(len(joint_distribution), dtype=int)
    # print(max_order_vector)

    x_lookup, w_lookup = _construct_lookup(
        orders=max_order_vector,
        dists=joint_distribution
        
    )

    return x_lookup, w_lookup

def generate_candidates(index_set, P):
    global candidates, pre_candidates, back_neighbours

    pre_candidates = []
    candidates = []
    vectors = np.identity(len(joint_distribution), dtype='int')

    for j in range(0, len(joint_distribution)):
        pre_candidates.append(index_set + vectors[j])

    for candidate in pre_candidates:
        back_neighbours = []
        for j in range(0, len(joint_distribution)):
            back_neighbour = candidate - vectors[j]
            if np.all((back_neighbour > 1)):
                back_neighbours.append(tuple(back_neighbour))

        if np.all([neighbour in old for neighbour in back_neighbours]):
            candidates.append(tuple(candidate))

    temp = []
    for candidate in candidates:
        # if candidate not in old:

         if np.all(np.array(candidate) <= P):# and np.linalg.norm(np.array(candidate),ord=1) <= (P+6):

            temp.append(candidate)

    candidates = temp
    
    for candidate in candidates:
        if candidate in old:
            candidates.remove(candidate)

    temp = []
    maxx = sum(np.max(np.array(old), axis=0))

    for candidate in candidates:

        if sum(np.max(np.array(old + [candidate]), axis=0)) > maxx:
            temp.append(candidate)

    candidates = temp

    return candidates


def sobol_error(vec):
    return np.linalg.norm(gt - vec) / gt_norm



def assign_errors(active_set):
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
            if np.isnan(poly[-1](*node)):
                poly_eval = 0
            else:
                poly_eval = poly[-1](*node)

            if node in dick.keys():
                # a, b, d, g = node
                current_errors.append(abs(dick[node] - poly_eval))

            else:
                print('Node not found')


                # current_errors.append(abs(solution - poly_eval))
                # dick[node] = solution

        active_errors.append(np.mean(current_errors))

    active = sorted(list(zip(active_set, active_errors)), key=lambda x: x[1])
    for i in active:
        if step > 1:
            if i[0] in old:
                active.remove(i)

    return active

  
def algorithm(P, dimension, joint_distribution,model, TOL, merge, parameter_names):
    
    global dick, old, candidates, poly, active, global_errors, no_nodes,step
    
    '''Initialise'''
    
    
    expansion = ch.generate_expansion(P, joint_distribution, normed = True)
    exponents = ch.lead_exponent(expansion, graded=True)
    vectors = np.identity(len(joint_distribution), dtype='int')

    date_today = datetime.date.today()
    
    start_time = time.perf_counter()

    step = 0

    old = [tuple(np.ones(len(joint_distribution),dtype='int'))]
    active = []
    poly = []
    uhats = []
    
    local_errors = []
    global_errors = []
    means = []
    
    names = parameter_names[0:dimension]
    
    df = pd.DataFrame(columns=['chosen_index','local_error','global_error','no_nodes','run_time'])
    df_indices = pd.DataFrame(columns=names,dtype=object)
    df_indices_s1 = pd.DataFrame(columns=names,dtype=object)

    
    '''Execute zeroth step'''
    
    # trivial = [seed]
    number_nodes,uhat = solver(old,expansion)
    uhats.append(uhat)
    assign_errors(old)
    
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
        
        number_nodes,uhat = solver(old,expansion)
        uhats.append(uhat)
        
        candidates = generate_candidates(chosen_index,P)
        stripped_active = [i[0] for i in active] + [j for j in candidates]
        active = assign_errors(stripped_active)
        
        sobol_time = time.perf_counter() 

        st = sense_t(uhat,exponents,expansion)
        s1 = sense_main(uhat,exponents,expansion)
        means.append(uhat[0])
        
        # print('Sobol time >>>', time.perf_counter() - sobol_time)
        

        global_errors.append(sobol_error(st))

        print("Global error >>>", global_errors[-1])

        """Save data"""
        run_time = time.perf_counter() - start_time

        
        numpoly.savez(f'../data/dimension_{dimension}/poly/poly_{P}+{date_today}.npz',*poly)
        np.savez(f'../data/dimension_{dimension}/poly/uhat_{P}+{date_today}.npz',*uhats)
        
        df_indices = df_indices.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3], 'e': st[4], 'f':st[5],'h':st[6]}, ignore_index=True)
        df_indices_s1 = df_indices_s1.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3], 'e': st[4], 'f':st[5],'h':st[6]}, ignore_index=True)
        df = df.append({'chosen_index': chosen_index,'local_error':local_errors[-1],                               'global_error':global_errors[-1],'no_nodes':number_nodes, 'run_time':run_time}, ignore_index=True)
        
        df.to_csv(f'../data/dimension_{dimension}/indices/run_file_{P}+{date_today}.csv')
        df_indices.to_csv(f'../data/dimension_{dimension}/indices/total_order_indices_{P}+{date_today}.csv')
        df_indices_s1.to_csv(f'../data/dimension_{dimension}/indices/first_order_indices_{P}+{date_today}.csv')

        print('Step time >>>', time.perf_counter() - start_time, 'seconds')
        print('-'*10,'break','-'*10)
        
        step += 1
        
    print('Congratulations, the algorithm has converged!')
    print('Here are the results...')
    print('-'*20)
    # print(f'ST_alpha:{st[0].round(4)}, ST_beta:{st[1].round(4)}, ST_delta:{st[2].round(4)},\
            # ST_gamma:{st[3].round(4)}, ST_e:{st[4].round(4)}, ST_f:{st[5].round(4)}, ST_h:{st[6].round(4)}') 

    print([i for i in zip(names,gt)])
    print([i for i in zip(names,st)])
              
    # print(f'GT_alpha:{gt[0].round(4)}, GT_beta:{gt[1].round(4)}, GT_delta:{gt[2].round(4)}, GT_gamma:{gt[3].round(4)}, GT_e:{gt[4].round(4)}, GT_f:{gt[5].round(4)}, GT_h:{gt[6].round(4)} ')
    print(f'The final grid contains {number_nodes} nodes.')
    print(f'The total run time was {df.run_time.sum()}seconds, not bad!')
    
    # if merge:
    
        # merged_set = merge_sets(old,active)
        
        # number_nodes,uhat = solver(merged_set, target, expansion)
            
        # st = sense_t(uhat,exponents,expansion)
        # s1 = sense_main(uhat,exponents,expansion)
        
        # print('-'*10,'MERGED','-'*10)
        # print(f'ST_alpha:{st[0].round(4)}, ST_beta:{st[1].round(4)}, ST_delta:{st[2].round(4)},\
            # ST_gamma:{st[3].round(4)}, ST_e:{st[4].round(4)}, ST_f:{st[5].round(4)}, ST_h:{st[6].round(4)}')                                    
              
        # print(f'GT_alpha:{gt[0].round(4)}, GT_beta:{gt[1].round(4)}, GT_delta:{gt[2].round(4)}, GT_gamma:{gt[3].round(4)}, GT_e:{gt[4].round(4)}, GT_f:{gt[5].round(4)}, GT_h:{gt[6].round(4)} ')
        
        # global_errors.append(sobol_error(st))


        # print('Global error >>>', global_errors[-1])
            
        # '''Save data'''
        # run_time = time.perf_counter() - start_time
     
        # numpoly.savez(f'../data/dimension_{dimension}/poly/poly_{P}+{date_today}.npz',*poly)
        # np.savez(f'../data/dimension_{dimension}/poly/uhat_{P}+{date_today}.npz',*uhats)
        
        # df_indices = df_indices.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3], 'e': st[4], 'f':st[5],'h':st[6]}, ignore_index=True)
        # df_indices_s1 = df_indices_s1.append({'alpha': st[0], 'beta': st[1], 'delta': st[2], 'gamma': st[3], 'e': st[4], 'f':st[5],'h':st[6]}, ignore_index=True)
        # df = df.append({'chosen_index': chosen_index,'local_error':local_errors[-1],                               'global_error':global_errors[-1],'no_nodes':number_nodes, 'run_time':run_time}, ignore_index=True)
        
        # df.to_csv(f'../data/dimension_{dimension}/indices/run_file_{P}+{date_today}.csv')
        # df_indices.to_csv(f'../data/dimension_{dimension}/indices/total_order_indices_{P}+{date_today}.csv')
        # df_indices_s1.to_csv(f'../data/dimension_{dimension}/indices/first_order_indices_{P}+{date_today}.csv')
        
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

    dim = len(joint_distribution)
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

    dim = len(joint_distribution)
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
def solver(old_set,expansion,current_names):



    global poly

    solver_time = time.perf_counter()
    
    vectors = np.identity(len(joint_distribution), dtype='int')

    nodes_list = []
    weights_list = []
    evals_list = []
    
    evals = {}
    transport = []
    
    # if np.all(old_set == np.zeros(len(joint_distribution))):
        # for index in old_set:
            # nodes, weights = build_nodes_weights(index)
            # nodes_list += nodes
            # weights_list += weights
        # evals_list = [0 for i in weights_list]
        # nodes_list = np.array(nodes_list).T
        # nodes_list.shape
        # len(evals_list)
        # polly, uhat = ch.fit_regression(expansion, nodes_list, evals_list, retall = 1)
        # poly.append(polly)
        # return len(weights_list),uhat
        
        

    for index in old_set:
        nodes, weights = build_nodes_weights(index)
        # weights = [weight * combinator(index,vectors) for weight in weights]

        nodes_list += nodes
        weights_list += weights

    
    for node in nodes_list:

        if node in run_dick.keys():
            evals_list.append(run_dick[node])
            
        elif node in dick.keys():
            evals_list.append(dick[node])

        else:
            transport.append(node)
            
            
    scenarios = [
        Scenario(name=None, **{k: v for k, v in zip(current_names, node)})
        for node in transport
    ]
    
    with MultiprocessingEvaluator(model, n_processes=10) as evaluator:
        results = evaluator.perform_experiments(scenarios)
    
    experiments, outcomes = results
    
    
    nodes = np.array(experiments[current_names])
    
    for node, outcome in zip(nodes, outcomes["fraction renewables"]):

        evals[tuple(node)] = outcome 
        dick[tuple(node)] = outcome
    
    
    abscissas = np.array(list(evals.keys())).T
    model_evals = list(evals.values())

    polly, uhat = ch.fit_regression(expansion, abscissas, model_evals, retall = 1)
    poly.append(polly)

    # print('Solver_time >>>', time.perf_counter() - solver_time)
    # print('Weight sum >>>', sum(weights_list))
    return len(weights_list), uhat



if __name__ == "__main__":


    parameter_names = [
        "progress ratio biomass",
        "progress ratio coal",
        "progress ratio hydro",
        "progress ratio igcc",
        "progress ratio ngcc",
        "progress ratio nuclear",
        "progress ratio pv",
        "progress ratio wind",
        "economic lifetime biomass",
        "economic lifetime coal",
        "economic lifetime gas",
        "economic lifetime hydro",
        "economic lifetime igcc",
        "economic lifetime ngcc",
        "economic lifetime nuclear",
        "economic lifetime pv",
        "economic lifetime wind",
    ]
    
    model_filename = "RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm"
    working_directory = "./models"
    
    dimension = 5
    expansion_order = 3
    merge=False
    
    joint_distribution,bounds = get_joint(5,parameter_names)

    model = setup_vensimmodel(model_filename, working_directory, parameter_names, bounds)
   
    gt,gt_norm = get_gt(dimension)
    run_dick = get_evals(dimension)
    dick = {}
    x_lookup, w_lookup = construct_wrapper(10,joint_distribution)
    
    old = [np.ones(len(joint_distribution),dtype='int')]
    expansion = ch.generate_expansion(expansion_order, joint_distribution, normed = True)
    current_names = parameter_names[0:dimension]
    
    number,uhat = solver(old,expansion,current_names)
    
    
    # algorithm(expansion_order, dimension, joint_distribution, model, 0.2, merge,parameter_names)
    

