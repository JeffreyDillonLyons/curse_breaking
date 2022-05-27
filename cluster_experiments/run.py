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
from ema_workbench import (
    ema_logging,
    MultiprocessingEvaluator,
    RealParameter,
    ScalarOutcome,
    Scenario,
    Samplers,
    save_results
)

from ema_workbench.em_framework import SobolSampler, get_SALib_problem

from SALib.analyze import sobol

# This is the script where my own sensitivity indices functions are kept
import sensitivity


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
    
def run_sobol(model_file, working_directory, parameter_names, resolution, dimension):
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
    
    Notes: 14:55 24/05/2022
    ---------------------------------
    I have edited this function so that the second-order index calculations
    are also saved, as well as the experiments and outcomes from ema_wrkbench.

    """
    current_names = parameter_names[0:dimension]

    with open("./data/variable_settings.json") as f:
        var_settings = json.loads(f.read())

    bounds = [var_settings[name] for name in current_names]

    model = setup_vensimmodel(model_file, working_directory, parameter_names, bounds)

    with MultiprocessingEvaluator(model, n_processes=50) as evaluator:
        results = evaluator.perform_experiments(
            resolution, uncertainty_sampling=Samplers.SOBOL
        )

    experiments, outcomes = results

    problem = get_SALib_problem(model.uncertainties)

    Si = sobol.analyze(problem, outcomes["fraction renewables"])
    
    params = [str(i) for i in range(1,dimension+1)]
    
    s2 = pd.DataFrame(columns=params*2)
    s2['params'] = params
    
    for i,ex in enumerate(Si['S2'].T):
        s2.iloc[:,i] = ex
    for i,ex in enumerate(Si['S2_conf'].T):
        s2.iloc[:,i+dimension] = ex
    
    s2 = s2.set_index(['params'])
    s2.to_csv(f"./data/sobol/sobol_s2_{dimension}_{resolution}.csv")
    
    Si.pop('S2')
    Si.pop('S2_conf')
    df = pd.DataFrame.from_dict(Si)
    df.to_csv(f"./data/sobol/sobol_{dimension}_{resolution}.csv")
    
    
    save_results(results, fr'./data/sobol/runfile_{dimension}_{resolution}.tar.gz')

    return results


def get_model_evals(
    model_file, working_directory, parameter_names, sparse, max_order, dimension
):
    """

    Parameters
    ----------
    model_file
    working_directory
    parameter_names
    sparse
    max_order
    dimension
  
    Returns
    -------
    
    Notes: 16:08 Thursday, 5 May 2022
    ---------------------------------
    As discussed it is better to use the cluster to complete the raw function evaluation.
    The polynomial fitting and inference can then be done later on my machine. This reduces
    the amount of debugging on your side if something goes awry. This function takes the
    desired dimensionality of the uncertainty space, as well as the maximum order of the 
    quadrature rule to form a grid and evaluate the model for the nodes in the grid. The
    output is then saved as a dictionary.
  

    """
    if sparse:
        grid = "sparse"

    else:
        grid = "gaussian"

    current_names = parameter_names[0:dimension]

    with open("./data/variable_settings.json") as f:
        var_settings = json.loads(f.read())

    bounds = [var_settings[name] for name in current_names]

    model = setup_vensimmodel(model_file, working_directory, current_names, bounds)

    distributions = [ch.Uniform(bound[0], bound[1]) for bound in bounds]

    joint_distribution = ch.J(*distributions)

    nodes = np.zeros((0, dimension), dtype="float")

    dick = {}

    for level in range(1, max_order + 1):
        n, w = ch.generate_quadrature(
            level, joint_distribution, sparse=sparse, rule="g", growth=False
        )
        nodes = np.concatenate((nodes, n.T), axis=0)

    scenarios = [
        Scenario(name="Jeff", **{k: v for k, v in zip(current_names, node)})
        for node in nodes
    ]

    with MultiprocessingEvaluator(model, n_processes=50) as evaluator:
        results = evaluator.perform_experiments(scenarios)

    experiments, outcomes = results

    nodes = np.array(experiments[current_names])

    for node, outcome in zip(nodes, outcomes["fraction renewables"]):
        dick[tuple(node)] = outcome

    dick = {str(k): float(v) for k, v in dick.items()}
    dump = json.dumps(dick)

    with open(f"./data/runs_dict_mo{max_order}_dim{dimension}_{grid}.json", "w") as f:
        f.write(dump)


def run_chaospy(
    model_file,
    working_directory,
    parameter_names,
    bounds,
    P,
    O,
    rule="g",
    sparse=False,
    growth=False,
    ID="gfg",
):
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
            print('True')
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

    scenarios = [
        Scenario(name=None, **{k: v for k, v in zip(parameter_names, node)})
        for node in transport
    ]

    with MultiprocessingEvaluator(model, n_processes=50) as evaluator:
        results = evaluator.perform_experiments(scenarios)

    experiments, outcomes = results

    nodes = np.array(experiments[parameter_names])

    for node, outcome in zip(nodes, outcomes["fraction_renewables"]):

        evals[
            tuple(node)
        ] = outcome  # potential problem here if the nodes are diff after returning from EMA >> weight_d
        dick[tuple(node)] = outcome

    abscissas = np.array(list(evals.keys())).T
    model_evals = list(evals.values())
    weights = [weight_d[i] for i in evals.keys()]

    if not len(abscissas.shape[0]) == len(weights):
        print("Weight_lookup failed")
        return

    polynomial, uhat = ch.fit_quadrature(
        expansion, abscissas, weights, model_evals, retall=1
    )

    poly_path = os.path.abspath(f"./data/polynomials_{ID}_poly_{O}_{P}_{today}.npz")
    uhat_path = os.path.abspath(f"./data/polynomials_{ID}_uhat_{O}_{P}_{today}.npz")
    index_path = os.path.abspath(f"./data/indices_{ID}_{O}_{P}_{today}.csv")

    with open(poly_path, "wb") as fh:
        numpoly.savez(fh, polynomial)
    with open(uhat_path, "wb") as fh:
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


def solver(model, results_file, joint_distribution, P, O, rule, sparse, growth, ID):
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

    with open(poly_path, "wb") as fh:
        numpoly.savez(fh, polynomial)
    with open(uhat_path, "wb") as fh:
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


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    """
    Here are all the parameters I have chosen, there are 17 in total here. As discussed, the first three campaings will be with 5,10,15 variables.  
    """
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

    '''
    Final SOBOL campaign - 12:48 24/05/2022
    > Amended code to also calculate and save second-order indices.
    ------------------------------------
    '''
    # ''' SOBOL ensemble runs for each dimension [5,8,10]'''
    
    #5 dimensions - 24,576 experiments
    run_sobol(
        model_filename, working_directory, parameter_names, resolution=2048, dimension=5
    )
    
    # 8 dimensions - 36,864 experiments
    run_sobol(
        model_filename, working_directory, parameter_names, resolution=2048, dimension=8
    )
    
    # 10 dimensions - 45,056 experiments
    run_sobol(
        model_filename, working_directory, parameter_names, resolution=2048, dimension=10
    )
    
    
    
