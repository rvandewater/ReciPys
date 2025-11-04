#!/bin/bash
#SBATCH --job-name=recipys_benchmarking
#SBATCH --account=sci-lippert
#SBATCH --partition=cpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=200gb
#SBATCH --output=logs/benchmark_%a_%j.log # %j is job id
#SBATCH --time=24:00:00
echo "Starting recipys benchmarking job"
eval "$(conda shell.bash hook)"

conda activate recipies
python benchmarking/perform_benchmark.py --seeds 1 2 3 4 5 --data_sizes 50 100 1000 10000 100000 1000000 #10000000 #100000000