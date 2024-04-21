#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=produce-batches
#SBATCH --error=../logs/produce-batches.err
#SBATCH --out=../logs/produce-batches.log

python produce-batches.py

