# Slurm script to run jax.distributed with older version of jax

The repository contains bash scripts that allow coordinator assignment for `jax.distributed` in a multi-host setting in Slurm. This is required only for codebases that rely on an old version of jax, as automatic assignment is supported in [the latest version](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html) of `jax.distributed` (see their [doc](https://jax.readthedocs.io/en/latest/multi_process.html) for setup).

## Tested environments
- jax 0.2.26, jaxlib 0.1.75+cuda11.cudnn82, driver CUDA version 12.0
 (Note: `slurm_distributed_test.py` also includes a fix to load cuda libraries manually for running an old version of jax on driver with cuda 12)

## Usage
Required parameters are `--file` which specifies the slurm job file to run in parallel and `--log` which specifies the logging directory. Any `coord_ip` file in the folder specified for `--log` will be **replaced** by a new file containing ip of the new coordinator node.

Read more about allowed input parameters and their default values by running
```
./slurm_dist.sh --help
```

`slurm_run.sh` needs to be modified for your own code execution: You should replace the first line below by an activation of your own conda environment, and `slurm_distributed_test.py` by your own python file.

```
source /PATH/TO/YOUR/CONDA/bin/activate YOUR_ENV
python slurm_distributed_test.py
```

To use `slurm_distributed_test.py` with the fix for jax 0.2.26, you should also replace the following line with path to the `lib` folder in your own conda environment:
```
LD_LIBRARY_PATH = '/PATH/TO/YOUR/CONDA/envs/YOUR_ENV/lib/'  # replace with path to `lib` for your conda env
```

## Testing
To test the code, you may run (after the above modifications for your environment):
```
./slurm_dist.sh --mem=5G --type=gpu --num_nodes=2 --num_devices=1 --port=1234 --timeout=1000 --file="slurm_run.sh" --log="slurm_log/" 
```

If the python script used is `slurm_distributed_test.py`, on success you should see the following output:
```
Initialised at 2024-03-13T18:54:05.105725.
[GpuDevice(id=0, process_index=0), GpuDevice(id=1, process_index=1)]
[GpuDevice(id=1, process_index=1)]
[2.]
```
Note that the time reported is UTC time.

## Troubleshoot
The following error may occur when GPU runs out of memory:
```
NCCL operation ncclGroupEnd() failed: invalid usage
```
The following output is expected as only gpus are used:
```
INFO:absl:Unable to initialize backend 'tpu_driver': NOT_FOUND: Unable to find driver in registry given worker: 
INFO:absl:Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
```