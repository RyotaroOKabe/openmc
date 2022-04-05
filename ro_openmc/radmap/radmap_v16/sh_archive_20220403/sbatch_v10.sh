#!/bin/bash
#SBATCH -J ryotaro
#SBATCH -A m3706
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -V
#SBATCH -q regular
#SBATCH -t 02:00:00

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=16
export HDF5_USE_FILE_LOCKING=FALSE
module load python
module load cray-hdf5-parallel
conda activate openmc-nersc
srun -n 4 -c 16 python radmap_uniform_mpi.py