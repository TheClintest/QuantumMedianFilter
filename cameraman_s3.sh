#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=350000MB
#SBATCH --account=cin_staff
#SBATCH --partition=g100_usr_interactive
#SBATCH --time=8:00:00
#SBATCH --error=cs3_err.out
#SBATCH --output=cs3.out
#SBATCH --job-name=qmfcs3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python main.py -g -mps -d cameraman_sp30 207 0.0001
