import jax.numpy as jnp
from jax import jit, random
import timeit

import jax 
print(jax.devices())

key = random.PRNGKey(0)
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)

from jax import device_put
import numpy as np
y = np.random.normal(size=(size, size)).astype(np.float32)
y = device_put(y)

def dot():
    jnp.dot(y, y.T).block_until_ready()  # runs on the GPU

t= timeit.timeit(dot, number=1)
print(t)
