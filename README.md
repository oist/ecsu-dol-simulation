# dol-simulation

Division of labor simulation based on joint tracking experiments.

## Alife 2021 paper
The paper **Evolution of Neural Complexity in Division of Labor Tasks** by **Ekaterina Sangati, Soheil Keshmiri, and Federico Sangati** is based on this code.

### Steps to reproduce the results:
1. Install `python 3.7.3` or above (https://www.python.org/)
2. Clone repository and checkout version tag `0.1.0`
   - `git clone https://gitlab.com/oist-ecsu/dol-simulation`
   - `cd dol-simulation`
   - `git checkout 0.1.0`
3. Create and activate python virtual environment, and upgrade pip
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `python -m pip install --upgrade pip wheel`
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
6. Alternatively, if you want to **run the simulation on a personal computer**: execute the `python3` command included in any slurm file above, setting seed and output directory appropriately.
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
