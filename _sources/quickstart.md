# Quickstart

Training MNIST on Caltech HPC H100 nodes.
See {doc}`the full walkthrough <examples/mnist-walkthrough>` for more detailed explanations.

## 1. Login to HPC

```bash
ssh <your-username>@login3.hpc.caltech.edu
```

## 2. Check the Python version

It is advisable to check the Python version that is loaded by default on login:

```
$ python --version
Python 3.9.18
```

If you are using codebases that depend on the latest version of deep learning
frameworks (`pytorch`, `jax`, etc.), chances are you will need to update to a
more modern Python version.
You can do so via HPC's module system, e.g.

```bash
$ module load python/3.10
$ python --version  # check loading worked as expected
Python 3.10.12
```

You can see all the available Python versions with:

```bash
$ module avail | grep "python/3\."
```

````{note}
You can ensure that the desired Python version is loaded by default by adding
the `module load` command to your `.bashrc`. For example, if you want to have
Python 3.11 available at login: `echo "module load python/3.11" >> .bashrc`.

This will save you from having to `module load` each time you log into HPC.
````

## 3. Download example code

```bash
mkdir repos && cd $_
git clone https://github.com/rossbar/pytorch-examples.git
```

## 4. Create virtual environment

```bash
mkdir -p ~/venvs
python3 -m venv ~/venvs/mnist-example
```

## 5. Install dependencies

```{note}
Pytorch must be installed on a compute node with GPU resources allocated.
See {doc}`the full walkthrough <examples/mnist-walkthrough>` for details.
```

```bash
# Request interactive terminal session on compute node w/ GPU(s)
srun --pty -t 00:30:00 --partition=gpu --gres=gpu:h100:1 -N 1 -n 1 /bin/bash -l

# Once on the compute node, activate the virtual environment and install dependencies
source ~/venvs/mnist-example/bin/activate
cd ~/repos/pytorch-examples/mnist
pip install -r requirements.txt ipython

# Test installation: if this raises and exception, something went wrong
python -c "import torch.nn as nn"

# Leave the interactive session on the compute node
exit
```

## 6. Set up paths

```bash
mkdir -p /central/groups/<your-research-group-name>/<your-username>/mnist_example
```

## 7. Prepare job script

Copy the following into a file called `train-mnist.sh`

```bash
#! /bin/bash

VENV=$HOME/venvs/mnist-example
SRCDIR=$HOME/repos/pytorch-examples/mnist
EXPERIMENT_DIR=/central/groups/<your-research-group-name>/<your-username>/mnist_example

# Activate Python virtual environment
source $VENV/bin/activate

# Sanity check: list out Python packages in current environment
pip list

# Run the job
python $SRCDIR/main.py \
  --data-path $EXPERIMENT_DIR \
  --save-model \
  --model-save-path $EXPERIMENT_DIR/model
```

## 8. Submit the job

```bash
chmod u+x train-mnist.sh
sbatch -t 00:10:00 --partition=gpu --gres=gpu:h100:1 -N 1 -n 8 train-mnist.sh
```

## 9. Monitor the job

Use `scontrol show job <job-id>` to poll the current status of the job.
The output (i.e. `stdout` and `stderr`) are piped to a file in your home
directory called `slurm-<job-id>.out`.
You can safely monitor this file as well using a read-only pager such as `less`,
even while the job is still running.

## 10. Check the results

Once the job is complete, you can verify that the trained model is at the
location specified in the job script:

```bash
# Activate virtual env on the login node
source ~/venvs/mnist-example/bin/activate
python
```

```python
>>> import torch
>>> model_path = "/central/groups/<your-research-group-name>/<your-username>/mnist_example/model/mnist_cnn.pt"
>>> # Remember - the login node is CPU-only and not for serious computation.
>>> model = torch.load(model_path, map_location=torch.device("cpu"))
>>> model.keys()
odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
```
