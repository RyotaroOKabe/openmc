#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH --time=15:00:00

export HDF5_USE_FILE_LOCKING=FALSE
module load python openmpi
module load cray-hdf5
conda activate openmc-train
srun -n 4 radmap_uniform_mpi.py