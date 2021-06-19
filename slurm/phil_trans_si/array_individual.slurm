#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --job-name=dot
#SBATCH --output=slurm_%A-%a.out
#SBATCH --mem=50G
#SBATCH --cpus-per-task=30
#SBATCH --array=1-20%20

# load python module
module load python/3.7.3
# module load ruse

# create a temporary directory for this job and save the name
seed_dir=${SLURM_JOB_ID}_`printf "%03d" ${SLURM_ARRAY_TASK_ID}`
tempdir=/flash/FroeseU/fede/${seed_dir}
mkdir ${tempdir}

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/dol-simulation
source env/bin/activate

# ruse
python3 -m dol.main \
--seed ${SLURM_ARRAY_TASK_ID} \
--dir $tempdir \
--gen_zfill True \
--num_pop 4,
--popsize 24 \
--max_gen 5000 \
--num_neurons 2 \
--num_trials 4 \
--num_random_pairings 3 \
--exclusive_motors_threshold 0.1 \
--switch_agents_motor_control True \
--cores 24

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir/* deigo:/bucket/FroeseU/fede/dol-simulation

# Clean up by removing our temporary directory
rm -r $tempdir