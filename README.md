Effortless Bayesian Deep Learning: Tapping into the Potential of Modern Optimizers
==============================

Bayesian methods allow for estimates of uncertainty which enable more efficient usage of data (i.e., via active learning) and avoid overfitting.
Furthermore, these uncertainty estimates improve model interpretability and assessment of model predictive confidence.

Currently, approximate Bayesian methods are either expensive to compute (MCMC), are significantly more difficult to implement (such as variational inference), or simply perform poorly and are limited in their Bayesian interpretation (MC dropout).
As such, there is demand for a method which exhibits the same computational cost as the optimization algorithms for deterministic neural networks while providing accurate posterior approximations and working out-of-the-box for any given architecture.

In this project, we strive to develop an effortless Bayesian method.
More specifically, this project investigates the connections between the Laplace approximation and the approximate second-order derivatives used in modern optimizers, such as Adam.
We use a sampling-based training procedure, where a sample is first drawn from a Gaussian weight-posterior, a gradient step is performed on this sampled neural network, and the variance of the weight-posterior is updated with an approximate Hessian.
Approximating the Hessian is the most time-consuming and painful-to-engineer step of this training procedure.
In this project, we tap into the potential of modern machine learning frameworks to efficiently approximate the Hessian.

# Development

## Setup

Clone and enter the repo by running:

```bash
git clone git@github.com:silasbrack/masters-thesis.git
cd masters-thesis
```

If you're running on the DTU LSF or are using environment modules, then run:

```bash
module load python3/3.9.11
module load cuda/11.6
```

Now you should be on the correct version of python and CUDA.
Now, create the python virtual environment by either running `make env` or the following script:
Note that if you don't want to develop new code, you can remove the `--extra dev` flag in the `pip-compile` command to not download unnecessary packages.

```bash
python3 -m venv venv/
source venv/bin/activate

python -m pip install --no-cache-dir -U pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

Restart the terminal.
Then, still within the `masters-thesis` folder, run:

```bash
source venv/bin/activate

wandb login
pre-commit install

cp .env.example .env
python src/data/download_data.py
```

## Submitting batch jobs

* Run `bsub < scripts/submit.sh` from within your LSF environment.

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

## Creating new optimizers

* https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
* https://pytorch.org/docs/stable/optim.html
* https://docs.pyro.ai/en/1.6.0/optimization.html
