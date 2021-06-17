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
- `perf`: the overall performance across multiple simulations (e.g., if an agent is undergoing x simulations with x other agents)
- `sim_perfs`: the list perfomances for each simulation
- `evo`: the `pyevolver.Evolution` object
- `sim`: the `dol.Simulation` object
- `sim_idx`: index of the simulation obtaining the best performance
- `data`: a list of dictionaries each containing the data from the n-th simulation. Each key in the dictionary maps to the data related to that key, e.g.:  
  - `target_position`: contains a list of `np.array` (one per simulation trial) representing the positions of the target.
  - `agents_brain_output`: contains a list of lists where `data[s]['agents_brain_output'][t][a]` represents the brain output of agent `a` of trial `t` of simulation `s`.




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
      > install.packages(c("dplyr", "tidyr", "car", "ggplot2", "ggsignif", "ggpubr", "pastecs", "compute.es"))
      > q() [quit R]
      Rscript dol_complexity.R
      ```
