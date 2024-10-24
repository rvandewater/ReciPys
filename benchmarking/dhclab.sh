#!/bin/bash
#SBATCH --job-name=yaib_experiment
#SBATCH --partition="cpu,gpupro,gpua100" # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=100gb
#SBATCH --output=classification_%a_%j.log # %j is job id
#SBATCH --gpus=0
#SBATCH --time=24:00:00

eval "$(conda shell.bash hook)"
conda activate recipys_polars
wandb agent --count 1 robinvandewater/yaib-experiments/"$1"