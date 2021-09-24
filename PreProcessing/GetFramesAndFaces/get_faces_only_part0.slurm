#!/bin/bash
#SBATCH-p gpu
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 4
python3.9 get_faces_only_part0.py
