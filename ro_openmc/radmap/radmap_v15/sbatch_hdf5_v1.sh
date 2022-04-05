#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4

module load python 
module load openmpi
module load cray-hdf5
export HDF5_USE_FILE_LOCKING=FALSE
conda activate openmc-train
srun -n 4 radmap_uniform_hdf5.py