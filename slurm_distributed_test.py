import os
import datetime
# ===================================================================== #
#    Fix for running an old version of jax on driver with cuda 12       |
#      - manually load cuda dependencies                                |
#      - do not preallocate memory                                      |
#                                                                       |
from ctypes import cdll
LD_LIBRARY_PATH = '/PATH/TO/YOUR/CONDA/envs/YOUR_ENV/lib/'  # replace with path to `lib` for your conda env
cdll.LoadLibrary(LD_LIBRARY_PATH + 'libcublas.so.11')
cdll.LoadLibrary(LD_LIBRARY_PATH + 'libcudart.so.11.0')
cdll.LoadLibrary(LD_LIBRARY_PATH + 'libcublasLt.so.11')
cdll.LoadLibrary(LD_LIBRARY_PATH + 'libcufft.so.10')
cdll.LoadLibrary(LD_LIBRARY_PATH + 'libcurand.so.10')
cdll.LoadLibrary(LD_LIBRARY_PATH + 'libcusolver.so.11')
cdll.LoadLibrary(LD_LIBRARY_PATH + 'libcusparse.so.11')
cdll.LoadLibrary(LD_LIBRARY_PATH + 'libcudnn.so.8')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
#                                                                       |
# ===================================================================== #

print('Initialised at ' +  datetime.datetime.utcnow().isoformat() + '.')

import sys
from absl import logging
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)


import jax
from utils_slurm.jax_dist import initialize as initialize_with_timeout

# monkey-patch of jax.distributed to allow for timeout
jax.distributed.initialize = initialize_with_timeout
jax.distributed.initialize(coordinator_address=os.environ['COORD_IP'].strip()+":"+os.environ['PORT'].strip(), 
                            num_processes=int(os.environ['NUM_JOBS']),
                            process_id=int(os.environ['SLURM_ARRAY_TASK_ID']),
                            initialization_timeout=int(os.environ['TIMEOUT']))

print(jax.devices())
print(jax.local_devices())

xs = jax.numpy.ones(jax.local_device_count())
output = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
print(output)
