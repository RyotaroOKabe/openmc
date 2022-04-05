#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=68
#SBATCH --constraint=knl

srun radmap_uniform_mpi.py
