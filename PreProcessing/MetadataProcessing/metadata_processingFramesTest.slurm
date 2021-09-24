#!/bin/bash
#SBATCH-p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
python3.8 metadata_processingFramesTest.py
