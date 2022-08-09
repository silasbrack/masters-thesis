module load python3/3.9.11
module load cuda/11.6
source venv/bin/activate

python -m debugpy --wait-for-client --listen 10.66.20.1:1143 src/train.py
