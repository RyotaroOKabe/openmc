#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=15:00:00
#SBATCH --constraint=<architecture>
#SBATCH --qos=debug
#SBATCH --account=ryotaro

# set up for problem & define any environment variables here

srun -n 4 -c 11 radmap_uniform_mpi.py

# perform any cleanup or short post-processing here