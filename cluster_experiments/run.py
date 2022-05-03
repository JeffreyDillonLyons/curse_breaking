import numpy as np
import chaospy as ch
import pandas as pd
import time
import json
import datetime
import numpoly
import os

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
from ema_workbench import ema_logging, MultiprocessingEvaluator, RealParameter, \
    Samplers, ScalarOutcome, Scenario

# This is the script where my own sensitivity indices functions are kept
import sensitivity


# model = load_model(
#     "./models/RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm")
# be_quiet()
# os.chdir("..") # loading model drops us into the model folder, this gets us back out.


# Sensitivity campaign -
# Here we carry out a sensitivity analysis campaign with SALib to get baseline sensitivity index values.
# Sampling: Saltelli sequence.
# Samples = 512*(D + 2) = 3584 points in R^5.

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


# Main experiment

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
    model = VensimModel('vensimmodel', wd=working_directory, model_file=model_file)

    uncertainties = []
    for name, bound in zip(parameter_names, bounds):
        lower, upper = bound
        uncertainty = RealParameter(name, lower, upper)
        uncertainties.append(uncertainty)
    model.uncertainties = uncertainties

    model.outcomes = [ScalarOutcome("fraction renewables", function=return_last)]

    return model


def run_sobol(model_file, working_directory, parameter_names, bounds, resolution):
    """

    Parameters
    ----------
    model_file
    working_directory
    parameter_names
    bounds
    resolution

    Returns
    -------

    """
    model = setup_vensimmodel(model_file, working_directory, parameter_names, bounds)

    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(resolution, uncertainty_sampling=Samplers.SOBOL)

    return results


def run_chaospy(model_file, working_directory, parameter_names, bounds,
                P, O, rule="g", sparse=False, growth=False, ID="gfg"):
    model = setup_vensimmodel(model_file, working_directory, parameter_names, bounds)

    distributions = [ch.Uniform(bound[0], bound[1]) for bound in bounds]
    joint_distribution = ch.J(*distributions)

    expansion = ch.generate_expansion(P, joint_distribution, normed=True)

    # Generate quadrature rule and weights of order 'O' from joint pdf.
    nodes, weights = ch.generate_quadrature(
        O, joint_distribution, rule=rule, sparse=sparse, growth=growth
    )

    evals = {}
    weight_d = {}
    transport = []
    dick = {}

    # Check dictionary for existing node/eval pairs
    for idx, node in enumerate(nodes.T):
        weight_d[tuple(node)] = weights[idx]

        if tuple(node) in dick.keys():
            evals[tuple(node)] = dick[tuple[node]]
        else:
            # if node is not key, append to transport for later evaluation
            transport.append(node)

    # Evaluate model for nodes in transport
    # This is where the model evaluations are happening so this is where the
    # code could be parallelised if needed. For reference - on single
    # thread of inteli3 evaluation takes o.4 seconds. The most expensive job
    # is a seventh orders quadrature rule which requires 32,768 evaluations
    # = 273 minutes.

    scenarios = [Scenario(name=None, **{k:v for k,v in zip(names, node)}) for node in
                 transport]

    with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(scenarios)

    return results


def solver(model, results_file, joint_distribution, P, O, rule, sparse,
           growth, ID):
    """
    This function carries out the experiment by constructing an expansion and
    a quadrature grid with weights. Nodes for which values are not present in
    dict(Dick) are appended to transport for evaluation with the model. After
    evaluation the nodes, weights, and evals are compiled to column vectors
    from the various dictionaries and the polynomial is fit with
    ch.fit_quadrature. The sensitivity indices are calculated with the
    functions in './sensitivity.py' from 'uhat' the fourier coefficients of
    the fitted PCE.
    """

    # global evals, abscissas, model_evals, weights

    # Initialise
    # FIXME why does loading a vensim model changes the current working directory?
    cwd = os.getcwd()
    load_model(model)
    os.chdir(cwd)

    be_quiet()

    # Generate raw PCE of order 'P' which is normalised.
    expansion = ch.generate_expansion(P, joint_distribution, normed=True)

    # Generate quadrature rule and weights of order 'O' from joint pdf.
    nodes, weights = ch.generate_quadrature(
        O, joint_distribution, rule=rule, sparse=sparse, growth=growth
    )

    start_time = time.perf_counter()
    today = datetime.date.today()

    evals = {}
    weight_d = {}
    transport = []
    dick = {}

    # Check dictionary for existing node/eval pairs
    for idx, node in enumerate(nodes.T):
        weight_d[tuple(node)] = weights[idx]

        if tuple(node) in dick.keys():
            evals[tuple(node)] = dick[tuple[node]]
        else:
            # if node is not key, append to transport for later evaluation
            transport.append(node)

    # Evaluate model for nodes in transport
    # This is where the model evaluations are happening so this is where the
    # code could be parallelised if needed. For reference - on single
    # thread of inteli3 evaluation takes o.4 seconds. The most expensive job
    # is a seventh orders quadrature rule which requires 32,768 evaluations
    # = 273 minutes.

    for node in transport:
        # set the uncertain parameter values to the values in node.
        for name, parameter in zip(names, node):
            set_value(name, parameter)

        # run simulation
        run_simulation(results_file)

        try:
            model_eval = get_data(results_file, "fraction renewables")
        except IndexError:
            raise IndexError
        else:
            result = float(model_eval[-1])
            evals[tuple(node)] = result
            dick[tuple(node)] = result

    # Reform nodes/weights/evals column vectors >>> fit expansion
    abscissas = np.array(list(evals.keys())).T
    model_evals = list(evals.values())
    weights = [weight_d[i] for i in evals.keys()]

    polynomial, uhat = ch.fit_quadrature(
        expansion, abscissas, weights, model_evals, retall=1
    )

    poly_path = os.path.abspath(f"./data/polynomials_{ID}_poly_{O}_{P}_{today}.npz")
    uhat_path = os.path.abspath(f"./data/polynomials_{ID}_uhat_{O}_{P}_{today}.npz")
    index_path = os.path.abspath(f"./data/indices_{ID}_sobol_{O}_{P}_{today}.csv")

    with open(poly_path, 'wb') as fh:
        numpoly.savez(fh, polynomial)
    with open(uhat_path, 'wb') as fh:
        numpoly.savez(fh, uhat)

    run_time = time.perf_counter() - start_time
    no_samples = len(weights)

    # Calculate Sobol sensitivity indices
    s1 = sensitivity.sense_main(uhat, expansion, joint_distribution)
    st = sensitivity.sense_t(uhat, expansion, joint_distribution)

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
    with open("model_runs_dict.json", "w") as f:
        f.write(dump)

    return polynomial, s1, st, run_time


def wrapper():
    # This function just loops over the given polynomial expansion orders
    # and quadrature orders and calls solver for each. Code may be
    # parallised here, with each 'job' i.e solver(order,level etc...) sent
    # to a different core for evaluation.

    orders = [1, 2, 3, 4, 5, 6]
    levels = [1, 2, 3, 4, 5, 6, 7]

    for order in orders:
        for level in levels:
            solver(order, level, rule="g", sparse=False, growth=False, ID="gfg")
            # solver(order, level, rule='g', sparse = True, growth = False, ID='sg')
            # solver(order, level, rule='c', sparse = True, growth = True, ID='nsg')


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    '''
    Here are all the parameters I have chosen, there are 17 in total here. As discussed, the first three campaings will be with 5,10,15 variables.  
    '''
    names = [
        'progress ratio biomass'
        'progress ratio coal'
        'progress ratio hydro'
        'progress ratio igcc'
        'progress ratio ngcc'
        'progress ratio nuclear'
        'progress ratio pv'
        'progress ratio wind'
        'economic lifetime biomass'
        'economic lifetime coal'
        'economic lifetime gas'
        'economic lifetime hydro'
        'economic lifetime igcc'
        'economic lifetime ngcc'
        'economic lifetime nuclear'
        'economic lifetime pv'
        'economic lifetime wind'

    ]

    # Here we load the parameter value bounds from .json to a dictionary.
    # These are the bounds from 'energy_model.py' which you sent me.
    with open("./data/variable_settings.json") as f:
        var_settings = json.loads(f.read())

    # Here we instantiate the chaospy joint distribution by multiplying the
    # five uniform distributions together.
    bounds = [var_settings[name] for name in names]

    model_filename =  "RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm"
    working_directory = "./models"

    # results = run_sobol(model_filename, working_directory, names, bounds, 10)
    run_chaospy(model_filename, working_directory, names, bounds, 1, 1,)

    # distributions = [ch.Uniform(bound[0], bound[1]) for bound in bounds]
    # joint_distribution = ch.J(*distributions)
    #
    # # dictionary to in model evaluations will be stored with format > { tuple('node'): str(model_eval) }
    # model = os.path.abspath(
    #     "./models/RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm")
    # results_file = os.path.abspath("./models/Current.vdfx")
    #
    #
    # returns = solver(model, results_file, joint_distribution, 1, 1, rule="g",
    #        sparse=False, growth=False, ID="gfg")
    # print(returns)