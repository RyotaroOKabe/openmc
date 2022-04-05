#!/bin/bash
#SBATCH -J ryotaro
#SBATCH -A m3706
#SBATCH --constraint=haswell
#SBATCH --tasks-per-node=4
#SBATCH -V
#SBATCH -t 00:30:00

export HDF5_USE_FILE_LOCKING=FALSE
module load python
module load cray-hdf5-parallel
conda activate openmc-nersc
srun -n 4 -c 4 python radmap_train_discrete_v3.2.py