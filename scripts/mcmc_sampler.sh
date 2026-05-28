#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --account=desi
#SBATCH --output=/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mlokken@ifae.es


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load python
conda activate cupix
python /global/common/software/desi/users/mlokken/cupix/scripts/mcmc_sampler.py test /pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/fcast_best_fit_arinyo_from_p1d_theory_config.yaml /pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/inference_config_ex.yaml