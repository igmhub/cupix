#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --account=desi
#SBATCH --output=/pscratch/sd/m/mlokken/desi-lya/px/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/m/mlokken/desi-lya/px/logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mlokken@ifae.es

module load python
conda activate cupix
cd $PSCRATCH/desi-lya/px/
srun -n 128 -c 1 /global/common/software/desi/users/mlokken/cupix/scripts/chi2_scan_mock.py arinyo tight bias beta