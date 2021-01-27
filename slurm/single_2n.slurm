#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --job-name=dyadic_exp
#SBATCH --mem=500G
#SBATCH --cpus-per-task=128
##SBATCH --mail-type=ALL
##SBATCH --mail-user=federico.sangati2@oist.jp

# load python module
# module load ruse
module load python/3.7.3
# module load ruse

# select the seed
seed=1

# create a temporary directory for this job and save the name
seed_dir=${SLURM_JOB_ID}_`printf "%03d" ${seed}`
tempdir=/flash/FroeseU/fede/seed_${seed_dir}
mkdir ${tempdir}

# Start 'myprog' with input from bucket,
# and output to our temporary directory
cd ~/Code/dol-simulation
source .env/bin/activate

# ruse
python -m dol.main \
--dir $tempdir \
--cores 120 \
--seed ${seed} \
--num_neurons 2 \
--popsize 96 \
--num_gen 5000 \
--trial_duration 80

# copy our result back to Bucket. We use "scp" to copy the data 
# back  as bucket isn't writable directly from the compute nodes.
rsync -avq $tempdir/* deigo:/bucket/FroeseU/fede/dol-simulation

# Clean up by removing our temporary directory
rm -r $tempdir