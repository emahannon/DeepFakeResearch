#!/bin/bash
#SBATCH-p gpu
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 4
python3.8 metadata_processing5.py
