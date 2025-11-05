#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10-00:00:00
#SBATCH --output %x.%j.out
#SBATCH --account=chengcli1

module restore

source ~/.bashrc

conda deactivate
conda activate crane

python3 amars.py input_yaml.yaml
