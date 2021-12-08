#!/bin/bash
#SBATCH-p gpu
#SBATCH --gres=gpu:kepler:1
#SBATCH -c 4
python pyt_dfd_5loadVideo.py
