import numpy as np
import chaospy as ch
import pandas as pd
import time
import json
import datetime
import numpoly
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
    # Samplers
)

# from ema_workbench.em_framework import SobolSampler, get_SALib_problem

# from SALib.analyze import sobol

# This is the script where my own sensitivity indices functions are kept
import sensitivity

def get_evals(max_order,dimension,grid):

    path = f'.\\data\\runs_dict_mo{max_order}_dim{dimension}_{grid}.json'
    
    path = os.path.abspath(path)
    
    with open(path,'rb') as f:
        dick = json.loads(f.read())
        
    dick = {eval(k):v for k,v in dick.items()}
    
    return dick
    
    
def fit_polynomial(
    polynomial_order,
    level,
    dick,
    rule="g",
    sparse=False,
    growth=False,
):
    if sparse:
        ID = "sg"

    else:
        ID = "gfg"
        
    current_names = parameter_names[0:dimension]

    with open("./data/variable_settings.json") as f:
        var_settings = json.loads(f.read())

    bounds = [var_settings[name] for name in current_names]
    
    
    distributions = [ch.Uniform(bound[0], bound[1]) for bound in bounds]

    joint_distribution = ch.J(*distributions)
    
    expansion = ch.generate_expansion(polynomial_order, joint_distribution, normed=True)
    
    nodes, weights = ch.generate_quadrature(
        level, joint_distribution, rule=rule, sparse=sparse, growth=growth
    )
    
    evals = {}
    weight_d = {}
    start = time.perf_counter()
    
    for idx, node in enumerate(nodes.T):
        weight_d[tuple(node)] = weights[idx]
        
        if tuple(node) in dick.keys():
            evals[tuple(node)] = dick[tuple(node)]
            
        else:
            print(f'Node: {tuple(node)} not found')
        
        
    abscissas = np.array(list(evals.keys())).T
    model_evals = list(evals.values())
    weights = [weight_d[i] for i in evals.keys()]   

    polynomial, uhat = ch.fit_quadrature(
        expansion, abscissas, weights, model_evals, retall=1
    )        
    
    today = datetime.date.today()
    poly_path = os.path.abspath(f"./data/polynomials_{ID}_poly_{polynomial_order}_{level}_{today}.npz")
    uhat_path = os.path.abspath(f"./data/polynomials_{ID}_uhat_{polynomial_order}_{level}_{today}.npz")
    index_path = os.path.abspath(f"./data/indices/indices_{ID}_{polynomial_order}_{level}_{today}.csv")

    with open(poly_path, "wb") as fh:
        numpoly.savez(fh, polynomial)
    with open(uhat_path, "wb") as fh:
        numpoly.savez(fh, uhat)

    run_time = time.perf_counter() - start
    no_samples = len(weights)

    # Calculate Sobol sensitivity indices
    s1 = sensitivity.sense_main(uhat, expansion, joint_distribution)
    st = sensitivity.sense_t(uhat, expansion, joint_distribution)

    index = np.arange(len(current_names))
    df = pd.DataFrame(columns=["params", "S1", "ST"], index=index)

    for i, name in enumerate(current_names):
        df.loc[i, "params"] = current_names[i]

    df["S1"] = s1
    df["ST"] = st

    df["run_time"] = run_time
    df["no_samples"] = no_samples

    df.to_csv(index_path)
    
    return df
    
def plotter(grid):

    if grid == 'sparse':
        ID = 'sg'
    elif grid == 'gaussian':
        ID = 'gfg'
        
    files = os.listdir('./data/indices')
    path = os.path.abspath('.')
    results = {f'{i}':{} for i in range(1,7)}
    for file in files:
        path + '\data\indices\ ' + file
        if os.path.isfile(path + '\data\indices' + '\\' +file):
            if ID in file:
            
                order = file[12]
                
                if file[15]==0:
                    level = '10'
                else:
                    level = file[14]
                
                df = pd.read_csv(path + '\data\indices' + '\\' + file).ST
                results[order][level] = np.linalg.norm(df)
    print(results.keys())            
    fig,ax = plt.subplots()
    
    colors = ['paleturquoise','cornflowerblue','plum','orange','firebrick','green']
    markers = ['o','s','^','+','x','*']
    
    for idx,i in enumerate(results.values()):
        ax.plot(i.keys(),i.values(),marker=markers[idx],label = f'p = {idx + 1}', color = colors[idx],linewidth=2 ,markersize=7)
        
    plt.show()
        



    
                
                
                
        

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
    
    max_order = 10
    dimension = 4
    grid = 'sparse'
    
    dick = get_evals(max_order,dimension,grid)
    
    oss = [1,2,3,4,5,6]
    levels = [2,3,4,5,6,7,8,9,10]
    
    # for order in oss:
        # for level in levels:
            # fit_polynomial(order,level,dick=dick,sparse = True)
            # print('Job_complete')
            
    plotter('gaussian')
            
            
            
    
    

    

   