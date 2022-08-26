#!/bin/bash
#BSUB -q gpua100
#BSUB -J train
#BSUB -n 8
#BSUB -gpu "num=2"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "select[hosts=1]"
#BSUB -u s174433@student.dtu.dk
#BSUB -N
#BSUB -oo logs/train-%J.out
#BSUB -eo logs/train-%J.err

module load python3/3.9.11
module load cuda/11.6
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

python src/train.py -m 'experiment=glob(*)' trainer.gpus=2
