'''Preamble'''

import numpy as np
import matplotlib.pyplot as plt
import chaospy as ch
from scipy import integrate
import pandas as pd
from mpl_toolkits import mplot3d
import time
import json
import pickle
from IPython.display import Image
import pandas as pd
import datetime
import numpoly
from itertools import product
from collections import OrderedDict
from os import PathLike
import re
import os

#Chaospy
import chaospy as ch

#EMA Workbench
import ema_workbench as em

from ema_workbench import (RealParameter, TimeSeriesOutcome, ema_logging,
                           MultiprocessingEvaluator, ScalarOutcome,
                           perform_experiments, CategoricalParameter,
                           save_results, Policy, )
from ema_workbench.connectors.vensim import (VensimModel, load_model, set_value, run_simulation, 
                                             get_data, be_quiet, vensimDLLwrapper)
from ema_workbench.connectors.vensim import load_model
from ema_workbench.em_framework.evaluators import SequentialEvaluator

def c(a):
    os.chdir(a)
    return os.getcwd()

''' Initialise data structures'''


names = ['demand fuel price elasticity factor',         #names of paramters to vary
 'economic lifetime biomass',
 'economic lifetime coal',
 'economic lifetime gas',
 'economic lifetime igcc']

f = open('./data/variable_settings.json')               #read variable bounds from dictionary
var_settings = json.loads(f.read())

distributions = []

for name in names:
    bounds = var_settings[name]
    distributions.append(ch.Uniform(bounds[0],bounds[1])) #create uniform chaospy distribution for each uncertain variable.
    
joint = ch.J(*distributions)                              #create a joint input distributions by combining all variables.

dick = {}

def solver(P,O,rule,sparse,growth,ID):
    
    global model,evals
    
    assert isinstance(rule,str)
    
    ''' Initialise'''
    
    expansion = ch.generate_expansion(P, joint)           #generate PCE of required order

    nodes,weights = ch.generate_quadrature(O, joint, rule=rule,sparse=sparse, growth=growth)     #generate quadrature nodes and weights   
    
    start_time = time.perf_counter()
    
    today = datetime.date.today()
    
    evals = {}
    weight_d = {}
    transport = []         #this is where we store nodes for which we do not already have an evaluation, transport is then sent for evaluations.
    
    ''' Check dictionary for existing node/eval pairs'''
    
    for idx,node in enumerate(nodes.T):
        
        weight_d[tuple(node)] = weights[idx] # add nodes and weights to dictionary
        
        if tuple(node) in dick.keys():
            
            evals[tuple(node)] = dick[tuple[node]] #search dictionary for nodes, take eval value if node is a key.
        
        else:
            
            transport.append(node)  #if node is not key, append to transport for later evaluation
            
    '''Evaluate model for nodes in transport'''
    
    model = load_model('./models/RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm')
    be_quiet = {}
    
    for node in transport:
        
        for name,parameter in zip(names,node):
            
            set_value(name,parameter)       #set the uncertain parameter values to the values in node.
            
        run_file = f'./data/vensim/{time.perf_counter()}.vdf'
            
        vensimDLLwrapper.start_simulation(0,0,1)
            
        try:
            model_eval = get_data(run_file,'fraction renewables')[-1]
            evals[tuple(node)] = model_eval
            dick[tuple(node)] = model_eval
            vensimDLLwrapper.finish_simulation()
            
        except IndexError:
            print('Model failed for node >>>', node)
            vensimDLLwrapper.finish_simulation()
            #return
    
    ''' Reform nodes/weights/evals column vectors >>> fit expansion'''
        
    abscissas = evals.keys()                    
    model_evals = evals.values()
    weights = [weight_d[i] for i in abscissas]

    polynomial = ch.fit_quadrature(joint,abscissas,weights,model_evals)
    
    poly_path = f'./data/polynomials/{ID}_poly_{O}_{P}_{today}.npz'
    index_path = f'./data/indices/{ID}_sobol_{O}_{P}_{today}'
    
    polynomial.savez(poly_path)
    
    run_time = time.perf_counter() - start_time
    no_samples = len(weights)
    
    ''' Calculate Sobol sensitivity indices'''
    
    s1 = ch.Sens_m1(polynomial,joint)  #calculate sensitivity indices from polynomial
    st = ch.Sens_t(polynomial,joint)
    
    saver(P,O,s1,st)
    
    return polynomial,s1,st,run_time

def saver(order,level,s1,st):
    
    index = np.arange(len(names))
    df = pd.DataFrame(columns=['params','S1','ST'], index = index)

    for i,name in enumerate(names):
        df.loc[i,'params'] = names[i]
    
    df['S1'] = s1
    df['ST'] = st
    
    df['run_time'] = run_time
    df['no_samples'] = no_samples
    
    index_path = f'./data/indices/{ID}_sobol_{O}_{P}_{today}.csv'    
    df.to_csv(index_path)
    
    dump = json.dumps(dick)
    f = open('model_runs_dict.json','w')
    f.write(dump)
    f.close

    
def wrapper():
    
    orders = [1,2,3,4,5,6]
    levels = [1,2,3,4,5,6,7,8]
    
    for order in orders:
        
        for level in levels:
            
            solver(order, level, rule='g', sparse = False, growth=False, ID='gfg')
            solver(order, level, rule='g', sparse = True, growth = False, ID='sg')
            solver(order, level, rule='c', sparse = True, growth = True, ID='nsg')

wrapper()          
            
            

            
            