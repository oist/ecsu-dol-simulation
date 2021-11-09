# dol-simulation

Division of labor simulation based on joint tracking experiments.

## How to guide

### Main
Runs evolutionary code and the simulation. To get the full list of option run:

`python -m dol.main --help`

If you provide an output directory it will generate:
- a list of of files `evo_xxx.json` containing the parameters of `pyevolver.Evolution` object at generation `xxx` (including the genotype of the agents' population, and the fitness).
- a file `simulation.json` with the parameters of the `dol.Simulation` that the agents go throughout evolution.

### Rerun simulation
Assuming you have a specific dir with outputs from the `dol.main` code (see above), you can rerun the simulation of a given generation and a given agent in the population. 

For instance,

`python -m dol.run_from_dir --dir <dirpath>`

Will run the simulation of the last saved generation and the best agent in the simulation and

`python -m dol.run_from_dir --dir <dirpath> --write_data` will create a subfolder `data` in the same directory with all the data from the simulation. 

To get the full list of options run:

`python -m dol.run_from_dir --help`

You can also use this module inside a python script as follows:

```python
from dol.run_from_dir import run_simulation_from_dir

perf, sim_perfs, evo, sim, data, sim_idx = run_simulation_from_dir('path/to/data')
```

where:
- `perf`: the overall performance of the best agent across multiple simulations computed as the average of the performances across all the pairs of agents (`np.mean(sim_perfs)`).
- `sim_perfs`: the list of perfomances for each simulation (i.e., a list of `n` values, if an agent is undergoing `n` simulations with `n` other agents). Each performance is obtain as the average performances across trials (see `trials_performances` and `sim_performance` below). If the simulation is perfomed by a single gent `sim_perfs` is a list with a single value equaling `perf`.
- `evo`: the `pyevolver.Evolution` object.
- `sim`: the `dol.Simulation` object.
- `sim_idx`: index of the simulation obtaining the best performance (`np.argmin(sim_perfs)`).
- `data`: a list of dictionaries each containing the data from each of the `n` simulations (one per paired agent). In each dictionary contains the following keys.
   |key|Type/Shape|Description|
   |---|-----|-----------|
   |`trials_performances`| `list(num_trials)`  |list of performances for each trial between the two agents of the current simulation, computed as the average of the distance between `tracker`  and `target` within each trial (see `delta_tracker_target` below)
   `sim_performance`| `float`  |the average across trial performances (`np.mean(trials_performances)`.
   `current_agent_pop_idx`| `tuple(2)`  |a tuple `(pop_idx, p_idx)` where `pop_idx` is the index of the population of the current agent and `p_idx` the index of the current agent within the population.
   `paired_agent_pop_idx`|  `tuple(2)` |analogous to above for the paired agent. If the simulation is perfomed by a single agent this is `None`.
   `genotypes`| `ndarray` with size depending on the network architecture (e.g., number of neurons) |genotypes of the agent(s) in the current simulation.
   `phenotypes`|  `list` of 2 `dict` |phenotypes of the agent(s) in the current simulation.
   `genotype_distance`| `float`  |distance between the genotypes of the current agent and the paired one (`None` if there is only a single agent).  
   `delta_tracker_target`| `ndarray(num_trials, num_data_points)`  |a list of arrays (one per trial) each containing data points, one for each step of the simulation representing the distance between `tracker` and `target` at that step.
   `target_position`|  `ndarray(num_trials, num_data_points)` |contains a list of `np.array` (one per simulation trial) representing the positions of the target throughout the simulation.
   `tracker_position`| `ndarray(num_trials, num_data_points)`  |as above wrt target position.
   `tracker_angle`| `ndarray(num_trials, num_data_points)` |as above wrt target angle (only relevant in `2d` mode with `XY_MODE` disabled).
   `tracker_wheels`|  `ndarray(num_trials, num_data_points, num_motors)` |as above wrt tracker wheels.
   `tracker_velocity`| `ndarray(num_trials, num_data_points)`  |as above wrt tracker velocity (difference between the wheels).
   `tracker_signals`|  `ndarray(num_trials, num_data_points, num_sensors)` |as above wrt tracker signal.
   `agents_brain_output`| `ndarray(num_trials, num_agents, num_data_points, num_brain_neurons)`  |contains a list of arrays (one per simulation trial) where each array contains the brain output values in the corresponding trial.
   `agents_brain_input`| `ndarray(num_trials, num_agents, num_data_points, num_brain_neurons)` |as above, wrt agent brain input.
   `agents_sensors`| `ndarray(num_trials, num_agents, num_data_points, num_sensors)`  |as above, wrt agent sensors.  
   `agents_brain_state`| `ndarray(num_trials, num_agents, num_data_points, num_brain_neurons)` |as above, wrt agent brain state.
   `agents_derivatives`|   |as above, wrt agent brain derivatives.
   `agents_motors`| `ndarray(num_trials, num_agents, num_data_points, num_brain_neurons)` |as above, wrt agent motors.
   `agents_motors_control_indexes`| `list(num_trials)` of `tuple(2)`  |a list of tuples (one per trial) where in each tuple `(l,r)`, `l` is the index of the agent whose left output is controlling the left motor and `r` is the index of the agent whose right output is controlling the right motor.


## Alife 2021 paper
The paper **Evolution of Neural Complexity in Division of Labor Tasks** by **Ekaterina Sangati, Soheil Keshmiri, and Federico Sangati** is based on this code.

### Steps to reproduce the results:
1. Install `python 3.7.3`
2. Clone repository and checkout version tag `0.1.0`
   - `git clone https://github.com/oist/ecsu-dol-simulation`
   - `cd ecsu-dol-simulation`
   - `git checkout 0.1.0`
3. Create and activate python virtual environment, and upgrade pip
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `python -m pip install --upgrade pip`
4. Build required libraries
   - `pip install -r requirements.txt`
5. If you want to **run the simulations on a cluster**, execute the following 3 scritps in the `slurm` directory :
   - `sbatch slurm/array_2n_iso.slurm` (isolated condition)
   - `sbatch slurm/array_2n_switch.slurm` (generalist condition)
   - `sbatch slurm/array_2n_dual.slurm` (specialist condition)

   This will run each condition (`isolated`, `generalist`, `specialist`) on 20 seeds (20 times from seed `1` to `20`). The output directories are respectively: 
   - `1d_2n_exc-0.1_zfill`
   - `1d_2n_exc-0.1_zfill_rp-3_dual`
   - `1d_2n_exc-0.1_zfill_rp-3_switch`
   
   Our code has been run on 128 `AMD Epyc` CPUs nodes [cluster at OIST](https://groups.oist.jp/scs/deigo) running `CentOS 8`.
6. Alternatively, if you want to **run the simulation on a personal computer**: execute the `python3` command included in any slurm file above, setting `seed` and `output directory` appropriately.
7. Optionally, if you want to rerun a simulation of a given seed after running the simulation, and optionally visualize
animation and data plots of behavior and neural activity, run (see available arguments): `python -m dol.run_from_dir --help`
8. In order to obtain the analysis and plots in the paper, run the following commands: 
   - `python -m dol.alife21.run_analysis_and_plots --indir data --outdir data/analysis_alife21`, where `--indir` is the directory containing the data from the simultaions (`default=data`).
   - install `R` (https://www.r-project.org/) and run the following: 
      ```
      cd rstat
      R [enter into R]
      > install.packages(c("dplyr", "tidyr", "car", "ggplot2", "ggsignif", "ggpubr", "pastecs", "compute.es")
      > q() [quit R]
      Rscript dol_complexity.R
      ```
