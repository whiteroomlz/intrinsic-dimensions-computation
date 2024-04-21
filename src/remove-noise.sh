#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=produce-batches
#SBATCH --error=../logs/clean-msts.err
#SBATCH --out=../logs/clean-msts.log

module purge
module load Python

source deactivate
source activate venv

python remove-noise.py
