"""
    A modification of an old implementation of jax.distributed to incorporate timeout
    original source: https://github.com/google/jax/blob/3acbd44952b86f54de6c937d9ca0874e47b382f9/jax/_src/distributed.py
"""

import functools

from absl import logging
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client
from jax._src.lib import xla_extension

_service = None
def initialize(coordinator_address: str, num_processes: int, process_id: int,
              initialization_timeout: int = 300):
  """Initialize distributed system for topology discovery.

  Currently, calling ``initialize`` sets up the multi-host GPU backend, and
  is not required for CPU or TPU backends.

  Args:
    coordinator_address: IP address of the coordinator.
    num_processes: Number of processes.
    process_id: Id of the current processes.
    initialization_timeout: timeout for initialization, default to 300.

  Example:

  Suppose there are two GPU hosts, and host 0 is the designated coordinator
  with address '10.0.0.1:1234', to initialize the GPU cluster, run the
  following commands before anything else.

  On host 0
  >>> jax.distributed.initialize('10.0.0.1:1234', 2, 0)  # doctest: +SKIP

  On host 1
  >>> jax.distributed.initialize('10.0.0.1:1234', 2, 1)  # doctest: +SKIP
  """
  if process_id == 0:
    global _service
    assert _service is None, 'initialize should be called once only'
    logging.info('Starting JAX distributed service on %s', coordinator_address)
    _service = xla_extension.get_distributed_runtime_service(coordinator_address,
                                                             num_processes)

  client = xla_extension.get_distributed_runtime_client(coordinator_address,
                                                        process_id,
                                                        init_timeout=initialization_timeout)
  logging.info('Connecting to JAX distributed service on %s', coordinator_address)
  client.connect()

  factory = functools.partial(xla_client.make_gpu_client, client, process_id)
  xla_bridge.register_backend_factory('gpu', factory, priority=300)