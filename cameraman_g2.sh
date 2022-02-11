#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=350000MB
#SBATCH --account=cin_staff
#SBATCH --partition=g100_usr_interactive
#SBATCH --time=8:00:00
#SBATCH --error=cg2_err.out
#SBATCH --output=cg2.out
#SBATCH --job-name=qmfcg2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python main.py -g -mps -d cameraman_g10 2.1 0.0001
