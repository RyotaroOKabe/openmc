#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH -q regular
#SBATCH --time=120

export HDF5_USE_FILE_LOCKING=FALSE
module load python
module load cray-hdf5-parallel
conda activate openmc-nersc
srun -n 4 python radmap_uniform_mpi.py