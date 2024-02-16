# Install pytorch

Full GPU support requires GPUs to be present at install time.
In other words, at least one GPU must be available at the point when you call
`pip install torch` (and torchvision, etc.)

There are two main ways to do so:
1. Set up environments interactively with `srun`.
2. Include the environment specification in the job script.

`````{tab-set}

````{tab-item} Interactively
```bash
# Request an interactive session on a compute node with GPU resources allocated, e.g.
srun --pty -t 00:00:30 --partition=gpu --gres=gpu:h100:1 -N 1 -n 1 /bin/bash -l

# Once the session has begun, activate the virtual environment you'd like to
# install pytorch into, e.g.
source ~/venvs/pytorch-dev/bin/activate
# You should now see (pytorch-dev) in your prompt string, indicating you are inside
# the desired environment

# Install the desired packages, e.g.
pip install torch torchvision pytorch3d

# Exit the interactive session on the compute node
exit
```
````

````{tab-item} Job script
Add lines to your job script to create an environment and install packages, e.g.

```bash
VENV=$HOME/venvs/pytorch
# Create a Python virtualenv for the job
python3 -m venv $VENV
# Activate environment
source $VENV/bin/activate
# Install necessary dependencies
pip install torch torchvision pytorch3d
```

The first time you submit this job, the environment will be created and packages
installed. Subsequent submissions should skip these steps without errors
because the environment already exists and contains all necessary packages.
````

`````
