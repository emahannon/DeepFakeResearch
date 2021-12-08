#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
python pyt_dfd_6.2loadVideo.py
