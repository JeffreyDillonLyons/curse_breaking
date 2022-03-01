'''
Created on 27 Jan 2014

@author: jhkwakkel
'''
from ema_workbench import (RealParameter, TimeSeriesOutcome, ema_logging,
                           MultiprocessingEvaluator, ScalarOutcome,
                           perform_experiments, CategoricalParameter,
                           save_results, Policy)
from ema_workbench.connectors.vensim import VensimModel
from ema_workbench.em_framework.evaluators import SequentialEvaluator


def get_energy_model():
    model = VensimModel('energyTransition', wd='./models',
                        model_file='RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm')
    
    model.outcomes = [
                      # TimeSeriesOutcome('cumulative carbon emissions'),
                      # TimeSeriesOutcome('carbon emissions reduction fraction'),
                      TimeSeriesOutcome('fraction renewables'),
                      # TimeSeriesOutcome('average total costs'),
                      # TimeSeriesOutcome('total costs of electricity'),
                      ]

        
    model.uncertainties = [RealParameter("year", 0.9, 1.1),                    
                         RealParameter("demand fuel price elasticity factor", 0, 0.5),                               
                         RealParameter("economic lifetime biomass", 30, 50),
                         RealParameter("economic lifetime coal", 30, 50),
                         RealParameter("economic lifetime gas", 25, 40),
                         RealParameter("economic lifetime igcc", 30, 50),
                         RealParameter("economic lifetime ngcc", 25, 40),
                         RealParameter("economic lifetime nuclear", 50, 70),
                         RealParameter("economic lifetime pv", 20, 30),
                         RealParameter("economic lifetime wind", 20, 30),
                         RealParameter("economic lifetime hydro", 50, 70,),               
                         RealParameter("uncertainty initial gross fuel costs", 0.5, 1.5),            
                         RealParameter("investment proportionality constant", 0.5, 4),
                         RealParameter("investors desired excess capacity investment", 0.2,2),                  
                         RealParameter("price demand elasticity factor", -0.07, -0.001),               
                         RealParameter("price volatility global resource markets", 0.1, 0.2),                      
                         RealParameter("progress ratio biomass", 0.85,1),
                         RealParameter("progress ratio coal", 0.9,1.05),
                         RealParameter("progress ratio gas", 0.85,1),
                         RealParameter("progress ratio igcc", 0.9,1.05),
                         RealParameter("progress ratio ngcc", 0.85,1),
                         RealParameter("progress ratio nuclear", 0.9,1.05),
                         RealParameter("progress ratio pv", 0.75,0.9),
                         RealParameter("progress ratio wind", 0.85,1),
                         RealParameter("progress ratio hydro", 0.9,1.05),                                                    
                         RealParameter("starting construction time", 0.1,3),                    
                         RealParameter("time of nuclear power plant ban", 2013,2100),
                         RealParameter("weight factor carbon abatement", 1,10),
                         RealParameter("weight factor marginal investment costs", 1,10),    
                         RealParameter("weight factor technological familiarity", 1,10),    
                         RealParameter("weight factor technological growth potential", 1,10),                  
                         RealParameter("maximum battery storage uncertainty constant", 0.2,3),
                         RealParameter("maximum no storage penetration rate wind", 0.2,0.6),
                         RealParameter("maximum no storage penetration rate pv", 0.1,0.4),
                         CategoricalParameter("SWITCH lookup curve TGC", (1,2,3,4)),
                         CategoricalParameter("SWTICH preference carbon curve", (1,2)),
                         CategoricalParameter("SWITCH economic growth", (1,2,3,4,5,6)), 
                         CategoricalParameter("SWITCH electrification rate", (1,2,3,4,5,6)),               
                         CategoricalParameter("SWITCH Market price determination", (1,2)),
                         CategoricalParameter("SWITCH physical limits", (1,2)), 
                         CategoricalParameter("SWITCH low reserve margin price markup", (1,2,3,4)), 
                         CategoricalParameter("SWITCH interconnection capacity expansion", (1,2,3,4)), 
                         CategoricalParameter("SWITCH storage for intermittent supply", (1,2,3,4,5,6,7)), 
                         CategoricalParameter("SWITCH carbon cap", (1,2,3), ),  
                         CategoricalParameter("SWITCH TGC obligation curve", (1,2,3)), 
                         CategoricalParameter("SWITCH carbon price determination", (1,2,3)),                      
                          ]   
    return model    


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)    
    model = get_energy_model()

    policies = [
        # Policy('adaptive policy', model_file='RB_V25_ets_1_policy_modified_adaptive_extended_outcomes.vpm'),
                Policy('no policy', model_file='RB_V25_ets_1_extended_outcomes.vpm')]
    n = 10000
    with MultiprocessingEvaluator(model, n_processes=55) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(n, policies=policies)

    outcomes.pop("TIME")
    results = experiments, outcomes
    save_results(results, f'./data/{n}_lhs.tar.gz')