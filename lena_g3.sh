#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=350000MB
#SBATCH --account=cin_staff
#SBATCH --partition=g100_usr_interactive
#SBATCH --time=8:00:00
#SBATCH --error=lg3_err.out
#SBATCH --output=lg3.out
#SBATCH --job-name=qmflg3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python main.py -g -mps -d lena_g15 1.9 0.0001
