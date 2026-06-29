#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cpu
#SBATCH --account=desi
#SBATCH --output=/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mlokken@ifae.es
#SBATCH --mem=8G

module load python
conda activate cupix
for z in {1..3}; do
    python /global/common/software/desi/users/mlokken/cupix/scripts/minimizer_pipeline.py hi_snr_drop_dla /pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/setup_config_mini_z${z}.yaml /pscratch/sd/m/mlokken/desi-lya/px/dr2_analysis/loa/inference_config_mini.yaml
done