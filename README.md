Master's Thesis
==============================

# Setup

## Quickstart

```bash
git clone git@github.com:silasbrack/masters-thesis.git
cd masters-thesis

module load python3/3.9.11
module load cuda/11.6
python3 -m venv venv/
source venv/bin/activate

python -m pip install --no-cache-dir -U pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

Restart terminal.

```bash
wandb login
pre-commit install

echo "DATA_DIR=\"data/\"" > .env
python src/data/download_data.py
```

## Submitting batch jobs

* Run `bsub < scripts/submit.sh`

## Debugging on GPU

* Install debugpy via `python -m pip install debugpy`
* If you're in Visual Studio Code, create a file under `.vscode/launch.json` with the following contents:
```json
{
    "name": "Python: Debug on GPU",
    "type": "python",
    "request": "attach",
    "connect": {
        "host": <GPU_IP>,
        "port": <PORT>,
    }
}
```
where `<GPU_IP>` is the IP address of your GPU (e.g., `10.66.20.1`) and `<PORT>` is the port that the debugger is listening on (e.g., `1143`).

Then, from your terminal on the GPU, run `python -m debugpy --wait-for-client --listen <GPU_IP>:<PORT> <FILE>`.
