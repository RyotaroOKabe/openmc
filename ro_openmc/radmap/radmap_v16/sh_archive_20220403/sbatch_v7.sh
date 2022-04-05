#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4

#this needs to be set when reading or writing files from $HOME or $CFS
export HDF5_USE_FILE_LOCKING=FALSE

#we don't need openmpi, let's use the default cray mpich
#module load python openmpi
module load python
#let's just use cray-hdf5-parallel since that's what we used to build
#module load cray-hdf5
module load cray-hdf5-parallel
conda activate openmc-train
#we need to add a python in front of our python script
#srun -n 4 radmap_uniform_mpi.py
srun -n 4 python radmap_uniform_mpi.py
