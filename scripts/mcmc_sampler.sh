#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH --account=desi
#SBATCH --output=/pscratch/sd/m/mlokken/desi-lya/px/logs/mcmc%x-%j.out
#SBATCH --error=/pscratch/sd/m/mlokken/desi-lya/px/logs/mcmc%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mlokken@ifae.es

module load python
conda activate cupix
python /global/common/software/desi/users/mlokken/cupix/scripts/mcmc_sampler.py