# dol-simulation

Division of labor simulation based on joint tracking experiments.

## Alife 2021 paper
The paper **Evolution of Neural Complexity in Division of Labor Tasks** by **Ekaterina Sangati, Soheil Keshmiri, and Federico Sangati** is based on this code.

### Steps to reproduce the results:
1. Install `python 3.7.3` (or above)
2. Clone repository and checkout version tag `0.1.0`
   - `git clone https://gitlab.com/oist-ecsu/dol-simulation`
   - `cd dol-simulation`
   - `git checkout 0.1.0`
3. Create and activate python virtual environment, and upgrade pip
   - `python3 -m venv .venv`
   - `source .venve/bin/activate`
   - `python -m pip install --upgrade pip wheel`
4. Build required libraries
   - `pip install -r requirements.txt`
5. If you want to **run the code on a cluster**, execute the following 3 scritps in the `slurm` directory :
   - `sbatch slurm/array_2n_iso.slurm` (isolated condition)
   - `sbatch slurm/array_2n_switch.slurm` (generalist condition)
   - `sbatch slurm/array_2n_dual.slurm` (specialist condition)

   This will run each condition 20 times on 20 seeds (`1` to `20`).
   Our code has been run on 128 `AMD Epyc` CPUs nodes [cluster at OIST](https://groups.oist.jp/scs/deigo) running `CentOS 8`.
6. To **run the code on a personal computer**: execute the `python3` command included in any slurm file above with appropriate arguments.
