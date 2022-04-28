import numpy as np
import matplotlib.pyplot as plt
import chaospy as ch
import pandas as pd
import time
import json
import datetime
import numpoly
import os
from SALib.sample import saltelli
from SALib.analyze import sobol
from ema_workbench import (
    RealParameter,
    TimeSeriesOutcome,
    ema_logging,
    MultiprocessingEvaluator,
    ScalarOutcome,
    perform_experiments,
    CategoricalParameter,
    save_results,
    Policy,
)

from ema_workbench.connectors.vensim import (
    VensimModel,
    load_model,
    set_value,
    run_simulation,
    get_data,
    be_quiet,
    vensimDLLwrapper,
)

# This is the script where my own sensitivity indices functions are kept
import sensitivity

# Here are the five parameters which I have chosen for the first sensitivity campaign.
names = [
    "progress ratio biomass",
    "progress ratio coal",
    "progress ratio nuclear",
    "progress ratio pv",
    "progress ratio wind",
]

#Here we load the parameter value bounds from .json to a dictionary. 
#These are the bounds from 'energy_model.py' which you sent me. 
with open("./data/variable_settings.json") as f: var_settings = json.loads(f.read())

bounds = [var_settings[name] for name in names]
 
#Here we instantiate the chaospy joint distribution by multiplying the five uniform distributions together. 
distributions = []

for bound in bounds:
    distributions.append(ch.Uniform(bound[0], bound[1]))

joint = ch.J(*distributions)

#dictionary to in model evaluations will be stored with format > { tuple('node'): str(model_eval) } 
dick = {}

run_file = (
    r"C:\Users\jeffr\OneDrive\Documents\Education\cluster_test\models\Adaptive5.vdfx"
)
model = load_model(
    "./models/RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm"
)
# be_quiet()
os.chdir("..") # loading model drops us into the model folder, this gets us back out. 


'''Sensitivity campaign - 
Here we carry out a sensitivity analysis campaign with SALib to get baseline sensitivity index values.
Sampling: Saltelli sequence.
Samples = 512*(D + 2) = 3584 points in R^5.
'''

# salib_problem = {"num_vars": 5, "names": names, "bounds": bounds}

# salib_samples = saltelli.sample(salib_problem, 512, calc_second_order=False)
# Y = np.zeros([salib_samples.shape[0]])

# for idx, node in enumerate(salib_samples):

    # for name, parameter in zip(names, node):

        # set_value(name, parameter)

    # run_simulation(run_file)
    
    # Y[idx] = float(get_data(run_file, "fraction renewables")[-1])


# Si = sobol.analyze(salib_problem, Y)

# Si.pop("S2")
# Si.pop("S2_conf")

# df_salib = pd.DataFrame.from_dict(Si)
# df_salib.to_csv("./data/salib.csv")


''' Main experiment
Thisfunction carries out the experiment by constructing an expansion and a quadrature grid with weights.
Nodes for which values are not present in dict(Dick) are appended to transport for evaluation with the model. 
After evaluation the nodes, weights, and evals are compiled to column vectors from the various dictionaries and the polynomial is fit with ch.fit_quadrature. The sensitivity indices are calculated with the functions in './sensitivity.py' from 'uhat' the fourier coefficients of the fitted PCE.
'''

def solver(P, O, rule, sparse, growth, ID):

    global model, evals, abscissas, model_evals, weights

    """ Initialise"""
    #Generate raw PCE of order 'P' which is normalised.
    expansion = ch.generate_expansion(
        P, joint, normed=True
    ) 
    #Generate quadrature rule and weights of order 'O' from joint pdf.
    nodes, weights = ch.generate_quadrature(
        O, joint, rule=rule, sparse=sparse, growth=growth
    )  

    start_time = time.perf_counter()

    today = datetime.date.today()

    evals = {}
    weight_d = {}
    transport = (
        []
    )  

    """ Check dictionary for existing node/eval pairs"""

    for idx, node in enumerate(nodes.T):

        weight_d[tuple(node)] = weights[idx]  

        if tuple(node) in dick.keys():

            evals[tuple(node)] = dick[
                tuple[node]
            ]  

        else:

            transport.append(       # if node is not key, append to transport for later evaluation
                node
            )  

    """Evaluate model for nodes in transport"""
    # This is where the model evaluations are happening so this is where the code could be parallelised if needed. 
    # For reference - on single thread of inteli3 evaluation takes o.4 seconds. The most expensive job is a seventh orders 
    # quadrature rule which requires 32,768 evaluations = 273 minutes.

    for node in transport:

        for name, parameter in zip(names, node):

            set_value(
                name, parameter
            )  # set the uncertain parameter values to the values in node.

        run_simulation(run_file)
        
        try:
            model_eval = float(get_data(run_file, "fraction renewables")[-1])
            evals[tuple(node)] = model_eval
            dick[tuple(node)] = model_eval
            
        except IndexError:
            pass
        
            

    """ Reform nodes/weights/evals column vectors >>> fit expansion"""

    abscissas = np.array(list(evals.keys())).T
    model_evals = list(evals.values())
    weights = [weight_d[i] for i in evals.keys()]

    polynomial, uhat = ch.fit_quadrature(
        expansion, abscissas, weights, model_evals, retall=1
    )

    poly_path = f"./data/polynomials/{ID}_poly_{O}_{P}_{today}.npz"
    uhat_path = f"./data/polynomials/{ID}_uhat_{O}_{P}_{today}.npz"
    index_path = f"./data/indices/{ID}_sobol_{O}_{P}_{today}.csv"

    numpoly.savez(poly_path, polynomial)
    np.savez(uhat_path, uhat)

    run_time = time.perf_counter() - start_time
    no_samples = len(weights)

    """ Calculate Sobol sensitivity indices"""
    s1 = sensitivity.sense_main(uhat, expansion,joint) 
    st = sensitivity.sense_t(uhat, expansion,joint)

    index = np.arange(len(names))
    df = pd.DataFrame(columns=["params", "S1", "ST"], index=index)

    for i, name in enumerate(names):
        df.loc[i, "params"] = names[i]

    df["S1"] = s1
    df["ST"] = st

    df["run_time"] = run_time
    df["no_samples"] = no_samples

    df.to_csv(index_path)
    save_dick = {str(key): value for key, value in dick.items()}
    dump = json.dumps(save_dick)
    f = open("model_runs_dict.json", "w")
    f.write(dump)
    f.close

    return polynomial, s1, st, run_time


#This function just loops over the given polynomial expansion orders and quadrature orders and calls solver for each.
#The code may be parallised here, with each 'job' i.e solver(order,level etc...) sent to a different cluster for evaluation. 


def wrapper():

    orders = [1,2,3,4,5,6]
    levels = [1,2,3,4,5,6,7]

    for order in orders:

        for level in levels:

            solver(order, level, rule="g", sparse=False, growth=False, ID="gfg")
            # solver(order, level, rule='g', sparse = True, growth = False, ID='sg')
            # solver(order, level, rule='c', sparse = True, growth = True, ID='nsg')


solver(1,1,rule="g", sparse=False, growth=False, ID="gfg")
