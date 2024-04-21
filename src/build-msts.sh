#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=720:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=build-msts
#SBATCH --error=../logs/build-msts-%A-%a.err
#SBATCH --out=../logs/build-msts-%A-%a.log
#SBATCH --array=1-100

idx=$SLURM_ARRAY_TASK_ID

python build-mst.py ../batches/batch-"$idx".npy
