# import matplotlib
# matplotlib.use('TkAgg')  # Use a backend that supports interactive display
import jax
import matplotlib.pyplot as plt
import numpy as np
import timeit
loops=1000

x_np = np.linspace(0, 10, 1000)
print(type(x_np))

y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np);
# plt.show()
plt.savefig('output.png')

import jax.numpy as jnp

x_jnp = jnp.linspace(0, 10, 1000)
print(type(x_jnp))

y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp);
plt.savefig('output2.png')

# NumPy: mutable arrays
x = np.arange(10)
x[0] = 10
print(x)

# jnp: immutable arrays trick
x = jnp.arange(10)
y = x.at[0].set(10)
print(x)
print(y)

import jax.numpy as jnp
jnp.add(1, 1.0)  # jax.numpy API implicitly promotes mixed types.

from jax import lax
lax.add(jnp.float32(1), 1.0)  # jax.lax API requires explicit type promotion.

x = jnp.array([1, 2, 1])
y = jnp.ones(10)
print(jnp.convolve(x, y))

from jax import lax
result = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy
print(result[0, 0])

import jax.numpy as jnp

def norm(X):
  X = X - X.mean(0)
  return X / X.std(0)

from jax import jit
norm_compiled = jit(norm)

np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
print(np.allclose(norm(X), norm_compiled(X), atol=1E-6))

def _norm_():
    norm(X).block_until_ready()

t = timeit.timeit(_norm_, number=loops)
print("{} ms per loop".format(t/loops*1000))

def _norm_compiled_():
    norm_compiled(X).block_until_ready()

t = timeit.timeit(_norm_compiled_, number=loops)
print("{} ms per loop".format(t/loops*1000))

def get_negatives(x):
  return x[x < 0]

x = jnp.array(np.random.randn(10))
print(get_negatives(x))

@jit
def f(x, y):
  print("Running f():")
  print(f"  x = {x}")
  print(f"  y = {y}")
  result = jnp.dot(x + 1, y + 1)
  print(f"  result = {result}")
  return result

x = np.random.randn(3, 4)
y = np.random.randn(4)
print(f(x, y))

from jax import make_jaxpr

def f(x, y):
  return jnp.dot(x + 1, y + 1)

print(make_jaxpr(f)(x, y))

from functools import partial

@partial(jit, static_argnums=(1,))
def f(x, neg):
  return -x if neg else x

print(f(12, True))

import jax.numpy as jnp
from jax import jit

@jit
def f(x):
  print(np.array(x.shape).prod())
  return x.reshape((np.prod(x.shape),))

x = jnp.ones((2, 3))
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 3))
print(x)
print(f(x))

@jit
def f(x):
  print(f"x = {x}")
  print(f"x.shape = {x.shape}")
  print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
  # comment this out to avoid the error:
  # return x.reshape(jnp.array(x.shape).prod())

f(x)