# Quickstart

An end-to-end example of setting up and running a model training job with
pytorch via [slurm][hpc-slurm].

Specifically, this quickstart will show how to setup and run a job training
a simple model on the ubiquitous [MNIST dataset][wiki-mnist] on one of HPC's
H100 nodes.

```{admonition} Before you begin...
:class: dropdown

Make sure you have an account with HPC.
See [the official documentation][hpc-account] for details.
```

## Logging on to HPC

First you must log in to the HPC cluster.
Details are provided in the [official documentation][hpc-login], with one
caveat: you must log in to *specific* login nodes in order to access the H100's,
either **login3** or **login4**:

```{code-block} bash
$ ssh <your-username>@login3.hpc.caltech.edu
```

Follow the instructions to complete the login with two-factor authentication.

After successful login, you will be on the *login node* inside the home directory
for your account:

```{code-block} bash
[<your-username>@login3 ~]$ pwd
/home/<your-username>
```

## Testing GPU access

The login node is not intended for computationally-intensive tasks, thus you will
not have direct access to GPU resources from the login node.

One way to do so is to log onto one of the compute nodes interactively using
`srun`:

```{code-block} bash
$ srun --pty -t 00:00:30 --partition=gpu --gres=gpu:h100:1 -N 1 -n 1 /bin/bash -l
```

As soon as your job is allocated, you should notice a change in the hostname
(the thing after the `@` in your bash prompt) indicating that you are now on
a compute node, e.g.

```{code-block} bash
$ srun --pty -t 00:00:30 --partition=gpu --gres=gpu:h100:1 -N 1 -n 1 /bin/bash -l
[<your-username>@hpc-33-13 ~]$
```

The compute node will have access to the gpu resources requested via the `--gres`
flag --- a single H100 card in this example.
From the compute node you can run familiar commands to query the state of the
GPU:

```{code-block} bash
[<your-username>@hpc-33-13 ~]$ nvidia-smi
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10              Driver Version: 535.86.10    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:E3:00.0 Off |                    0 |
| N/A   24C    P0              67W / 700W |      4MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

Type `exit` to exit the interactive terminal session on the compute node.
Notice that the hostname reverts back to `login3`, reflecting that you are
back on the login node.

```{code-block} bash
[<your-username>@hpc-33-13 ~]$ exit
logout
[<your-username>@login3 ~]$
```

## Setting up a training job

The ability to start interactive sessions on compute nodes via `srun` is a handy,
but it is not ideal for launching jobs.
Batch processing is a much better fit for real computational experiments - the
remainder of the quickstart shows how to setup, launch, and evaluate a model
training job with slurm.

### Downloading the example

We'll use the `mnist` example from a fork of the [`pytorch/examples`][gh-pt_examples]
for our example workflow.

The reason we're using a forked version is the example is better illustrate
best-practices when working on shared HPC systems: ensuring that data and other
components that require a lot of storage (e.g. saved models) are kept separate
from source code.

The original pytorch mnist example has hard-coded paths for saving data to the
source directory. We've [modified the example][gh-pt-fork-diff] to add additional
user flags to the run script allowing us to specify separate locations for
storing data and training outputs.

Begin by downloading the source code (if you haven't already):

```{caution}
Be sure you're on the login node for this part!
```

```{code-block} bash
$ mkdir repos && cd $_
$ git clone https://github.com/rossbar/pytorch-examples.git
```

### Setting up a virtual environment

For this workflow, we'll use the Python built-in `venv` module to manage
environments.

First we'll create a centralized location to keep our virtual environments:

```bash
$ mkdir -p ~/venvs
```

Then create the environment for this specific experiment. We'll call it
`mnist-example`:

```bash
$ python3 -m venv ~/venvs/mnist-example
```

Now enter the environment you've just created.
It should be empty, give-or-take libraries for packaging such as `pip`,
`setuptools` and/or `wheel`:

```bash
$ source ~/venvs/mnist-example/bin/activate
$ pip list
Package    Version
---------- -------
pip        21.2.3
setuptools 53.0.0
```

### Installing dependencies

This step requires careful attention when using libraries that interface with
GPUs, such as `pytorch`.
In order to fully support the hardware, GPUs *must* be discoverable at
installation time, i.e. when you run `pip install`.

**This means that you must install dependencies with device support (like `pytorch`)
on the compute nodes allocated GPU(s).**

There are multiple ways you can do so: one option is to include the dependency
installation step in your job script.
For illustrative purposes, we will instead create the environment interactively
using `srun`.

```{note}
Once an environment is created with the necessary dependencies properly
installed, it can be reused without any additional installation by simply
activating it again.
```

For clarity's sake, exit the `mnist-example` environment on the login node:

```bash
$ deactivate
```

Now, request an interactive terminal on a compute node with at least one GPU
allocated:

```bash
$ srun --pty -t 00:30:00 --partition=gpu --gres=gpu:h100:1 -N 1 -n 1 /bin/bash -l
```

Notice the increased wall time (`-t 00:30:00`) - it's good to give yourself
some leeway here in case the downloading/installation takes longer than expected.

```bash
$ source ~/venvs/mnist-example/bin/activate
$ cd ~/repos/pytorch-examples/mnist
$ pip install -r requirements.txt ipython
```

Once this has completed, you can test the successful installation:

```bash
$ python
>>> import torch.nn as nn
```

If the installation completed correctly, you shouldn't see any exceptions at
import time.
If you get an `ImportError` or `ModuleNotFoundError`, it means something went
wrong while installing pytorch.
Consider opening an issue to this repo!

Once the installation is complete, you can close your interactive session on
the compute node with `exit`.

### Preparing for the run

Now that the virtual environment is set up with all the packages need to run
our job, the focus shifts to making final preparations for the run.
In this case, this means ensuring that we have the dataset that we'll be
training on, and that all of the paths for loading/saving data have been
set up.

First, ensure you're on the login node, and activate the virtual environment

```{note}
:class: dropdown

Remember, once the environment is successfully created on the compute node,
it can be entered from anywhere (though GPU-specific features will only work
on nodes with GPU resources allocated to them!)
```

```bash
$ source ~/venvs/mnist-example/bin-activate
```

The mnist example provides a command-line interface using `argparse`, therefore
we can learn about the options by passing in the `--help` flag:

```
$ cd repos/pytorch-examples/mnist
$ python main.py --help
usage: main.py [-h] [--data-path DATA_PATH] [--batch-size N] [--test-batch-size N]
               [--epochs N] [--lr LR] [--gamma M] [--no-cuda] [--no-mps] [--dry-run]
               [--seed S] [--log-interval N] [--save-model]
               [--model-save-path MODEL_SAVE_PATH]

PyTorch MNIST Example

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        path to MNIST data (default: "../data")
  --batch-size N        input batch size for training (default: 64)
  --test-batch-size N   input batch size for testing (default: 1000)
  --epochs N            number of epochs to train (default: 14)
  --lr LR               learning rate (default: 1.0)
  --gamma M             Learning rate step gamma (default: 0.7)
  --no-cuda             disables CUDA training
  --no-mps              disables macOS GPU training
  --dry-run             quickly check a single pass
  --seed S              random seed (default: 1)
  --log-interval N      how many batches to wait before logging training status
  --save-model          For Saving the current Model
  --model-save-path MODEL_SAVE_PATH
                        path to which model will be saved (default: cwd)
```

The most important options are the `--data-path` and `--model-save-path` flags,
which determine where the training data and training output will be stored.
As a general rule of thumb, you should never store data in your home directory.
Your home directory on hpc is capped at 50GB by default, and is not intended
for storage/access of large data.
**As a rule of thumb - your home directory should only be used for source code
and environments**.
For more details on storage allocations on HPC, including central storage and
scratch space, see [the official documentation][hpc-storage].

For this simple example, the downloading of the training data is handled for
us by `torchvision.datasets`, so all we have to do is create a directory where
data will be stored.
Since the MNIST dataset is so small (~60MB) we don't have to worry about
storage utilization.
Go ahead and create a directory on your research group's central storage
partition (if you haven't already):

```bash
$ mkdir -p /central/groups/<your-research-group-name>/<your-username>/mnist_example
```

If your not sure what `<your-research-group-name>` is, try `ls /central/groups`
and see if there are any obvious candidates (e.g. your professor's last name).
Else ask the person from your group who gave you HPC access!

Now we have everything we need to run the job.

### Submitting the job

We'll use slurm's `sbatch` command to submit our job to the scheduler.
To do so, we first need to write the script that describes our job.
In our case, there are two main steps we need to include:
 1. Enter the virtual environment, and
 2. Run `main.py`

Create a new bash script somehwere in your home directory called `train-mnist.sh`
and paste the following into it:

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

Be sure to replace `<your-research-group-name>` and `<your-username>`.
`sbatch` should be smart enough to assume that whatever job script you pass it
is executable, but it's always good to be explicit:

```bash
chmod u+x train-mnist.sh
```

Now we're ready to submit the job.
This is a very small experiment, so we only set the wall time to 10 minutes and only
request a single GPU.

```bash
$ sbatch -t 00:10:00 --partition=gpu --gres=gpu:h100:1 -N 1 -n 8 train-mnist.sh
Submitted batch job <job-id>
```

```{note}
:class: dropdown

The `sbatch` options can be included in the job script directly with `#SBATCH`.
See [the official HPC docs](https://www.hpc.caltech.edu/documentation/slurm-commands)
for details.
```

`sbatch` returns the `<job-id>` of the submitted job.
The `<job-id>` can be used to look up the status of the job:

```bash
$ scontrol show job <job-id>
```

The status of the job (i.e. whether it's running, pending, has completed, or
failed) can be found in the `JobState` field.

`stdout` and `stderr` for batch jobs are piped to text files `slurm-<job-id>.out`
in your home directory.
You can monitor the output of your job with a file pager:

```bash
$ less slurm-<job-id>.out
```

### Checking the results

Assuming the job completed successfully[^1], there should be a model saved to
the path specified by `--model-save-dir`.
We can verify this from the login node:

```bash
$ source ~/venvs/mnist-example/bin/activate
$ ipython
```

```python
In [1]: import torch

In [2]: model_path = "/central/groups/<your-research-group-name>/<your-username>/mnist_example/model/mnist_cnn.pt"

In [3]: model = torch.load(model_path, map_location=torch.device("cpu"))

In [4]: model.keys()
Out[4]: odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
```

Of course, if you actually wanted to *use* the model for inference, you'd want
to set up another slurm job and run it on the compute nodes!

[hpc-slurm]: https://www.hpc.caltech.edu/documentation/slurm-commands
[hpc-account]: https://www.hpc.caltech.edu/documentation/account-information
[hpc-login]: https://www.hpc.caltech.edu/documentation/faq/how-do-i-login-cluster
[wiki-mnist]: https://en.wikipedia.org/wiki/MNIST_database
[gh-pt_examples]: https://github.com/pytorch/examples
[gh-pt-fork-diff]: https://github.com/pytorch/examples/compare/main...rossbar:pytorch-examples:add-path-flags
[hpc-storage]: https://www.hpc.caltech.edu/documentation/storage

[^1]: It should have! If it didn't, see what went wrong in the `slurm-<job-id>.out`
      log. If it was a problem with the tutorial, open an issue/PR!
