

from jax import make_jaxpr
import numpy as np


numpy_array = np.zeros((3,3), dtype=np.float32)
print("original array:")
print(numpy_array)

# In place, mutating update
numpy_array[1, :] = 1.0
print("updated array:")
print(numpy_array)

import sys

def custom_excepthook(type, value, traceback):
    print(f"Exception: {type.__name__}, {value}")
    # Customize the way you want to handle exceptions here

sys.excepthook = custom_excepthook

import jax.numpy as jnp

jax_array = jnp.zeros((3,3), dtype=jnp.float32)

# In place update of JAX's array will yield an error!
# jax_array[1, :] = 1.0

updated_array = jax_array.at[1, :].set(1.0)
print("original array unchanged:\n", jax_array)
print("updated array:\n", updated_array)

print("original array:")
jax_array = jnp.ones((5, 6))
print(jax_array)

new_jax_array = jax_array.at[::2, 3:].add(7.)
print("new array post-addition:")
print(new_jax_array)

print(jnp.arange(10)[12])

print(jnp.arange(10.0).at[11].get())

print(jnp.arange(10.0).at[90].get(mode='fill', fill_value=jnp.nan))

np.sum([1, 2, 3])

# jnp.sum([1, 2, 3])

def permissive_sum(x):
  return jnp.sum(jnp.array(x))

x = list(range(10))
print(permissive_sum(x))
print(make_jaxpr(permissive_sum)(x))
