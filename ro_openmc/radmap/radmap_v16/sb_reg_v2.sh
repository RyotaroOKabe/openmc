#!/bin/bash
#SBATCH -J ryotaro
#SBATCH -A m3706
#SBATCH --constraint=haswell
#SBATCH --tasks-per-node=4
#SBATCH -V
#SBATCH -q regular
#SBATCH -t 02:00:00

export HDF5_USE_FILE_LOCKING=FALSE
module load python
module load cray-hdf5-parallel
conda activate openmc-train
srun -n 4 -c 16 python radmap_uniform.py