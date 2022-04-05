#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH --time=15:00:00

module load python openmpi
module load cray-hdf5
conda activate openmc-train
srun -n 4 radmap_uniform_mpi.py