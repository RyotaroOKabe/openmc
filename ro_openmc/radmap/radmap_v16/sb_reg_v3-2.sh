#!/bin/bash
#SBATCH -J ryotaro
#SBATCH -A m3706
#SBATCH --constraint=haswell
#SBATCH --tasks-per-node=4
#SBATCH -V
#SBATCH -q regular
#SBATCH -t 01:00:00

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=4
export HDF5_USE_FILE_LOCKING=FALSE
module load python
module load cray-hdf5-parallel
conda activate openmc-train
srun -n 4 -c 4 python radmap_uniform.py