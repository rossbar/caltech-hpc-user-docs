# Build with CUDA

In order to compile CUDA, you will need to ensure `nvcc` is available.
The CUDA Toolkit is available in the `nvhpc` module:

```bash
module load nvhpc
```

To test that the module has loaded properly:

`````{tab-set}

````{tab-item} Interactive mode
```bash
srun --pty -t 00:00:30 --partition=gpu --gres=gpu:h100:1 -N 1 -n 1 /bin/bash -l

# Once the session on the compute node has begun:

nvcc --version
```
````

````{tab-item} Batch mode
```bash
echo "nvcc --version" > check-cuda.sh
chmod u+x check-cuda.sh
sbatch -t 00:10:00 --partition=gpu --gres=gpu:h100:1 -N 1 -n 8 check-cuda.sh
```

Once completed, `slurm-<jobid>.out` should contain the output of `nvcc --version`
````
`````
