#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=compute-dimensions
#SBATCH --error=../logs/compute-dimensions-%A-%a.err
#SBATCH --out=../logs/compute-dimensions-%A-%a.log
#SBATCH --array=1-2

idx=$SLURM_ARRAY_TASK_ID

python compute-dimensions-"$idx".py

