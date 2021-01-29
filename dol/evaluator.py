import os
from joblib import Parallel, delayed
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, List
from dol.simulation import Simulation
import numpy as np

'''
POPULATION EVALUATION
'''

@dataclass
class Evaluator:
    
    genotype_structure: Dict
    trial_duration: int    
    outdir: str
    num_cores: int

    def __post_init__(self):
        self.sim = Simulation(
            genotype_structure = self.genotype_structure,        
            trial_duration = self.trial_duration,  # the brain would iterate trial_duration/brain_step_size number of time            
            num_cores = self.num_cores     
        )

        if self.outdir is not None:      
            sim_config_json = os.path.join(
                self.outdir, 'simulation.json'
            )  
            self.sim.save_to_file(sim_config_json)


    def evaluate(self, population, rnd_seed):                
        
        population_size = len(population)

        if self.num_cores > 1:
            # run parallel job            
            sim_array = [Simulation(**asdict(self.sim)) for _ in range(self.num_cores)]
            performances = Parallel(n_jobs=self.num_cores)( 
                delayed(sim_array[i%self.num_cores].run_simulation)(population, i) \
                for i in range(population_size)
            )
        else:
            # single core
            performances = [
                self.sim.run_simulation(population, i)
                for i in range(population_size)
            ]


        return performances