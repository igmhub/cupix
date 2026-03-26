#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --constraint=cpu
#SBATCH --account=desi
#SBATCH --output=/pscratch/sd/m/mlokken/desi-lya/px/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/m/mlokken/desi-lya/px/logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mlokken@ifae.es

export OMP_NUM_THREADS=1

module load python
conda activate cupix
python /global/common/software/desi/users/mlokken/cupix/scripts/chi2_scan.py igm tight mF T0
