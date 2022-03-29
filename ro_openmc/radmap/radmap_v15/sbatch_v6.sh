#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH --time=15:00

module load python openmpi
conda activate openmc-env
mpiexec -n 4 -s 11 
python radmap_uniform_mpi.py