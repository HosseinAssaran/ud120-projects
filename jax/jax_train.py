import jax.numpy as jnp
from jax import grad, jit, random, vmap
import timeit
import jax 
loops=1000

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

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

z = random.normal(key, (1000000,))

selu_jit = jit(selu)

def _selu_():
    selu_jit(z).block_until_ready()

t = timeit.timeit(_selu_, number=loops)
print("{} ms per loop".format(t/loops*1000))

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.array([0.,1.,2.])
print(x_small)
derivative_fn = grad(sum_logistic)
print(sum_logistic(x_small))
print(derivative_fn(x_small))

print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

def first_finite_differences(f, x):
  eps = 1e-3
  return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])

print(first_finite_differences(sum_logistic, x_small))

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))


def apply_matrix(v):
  return jnp.dot(mat, v)

# print(mat)
# for v in batched_x:
#    print(v)
#    print(apply_matrix(v))

def naively_batched_apply_matrix(v_batched):
   return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')

def _naively_batched_apply_matrix_():
    naively_batched_apply_matrix(batched_x).block_until_ready()

print("natively shape of output:", naively_batched_apply_matrix(batched_x).block_until_ready().shape)
t = timeit.timeit(_naively_batched_apply_matrix_, number=loops)
print("{} ms per loop".format(t/loops*1000))

@jit
def batched_apply_matrix(v_batched):
  print(v_batched.shape)
  print(mat.T.shape)
  return jnp.dot(v_batched, mat.T)

def _batched_apply_matrix_():
   batched_apply_matrix(batched_x).block_until_ready()

print('Manually batched')
t = timeit.timeit(_batched_apply_matrix_, number=loops)
print("{} ms per loop".format(t/loops*1000))


@jit
def vmap_batched_apply_matrix(v_batched):
  return vmap(apply_matrix)(v_batched)

def _vmap_batched_apply_matrix_():
    vmap_batched_apply_matrix(batched_x).block_until_ready()

print('Auto-vectorized with vmap')
t = timeit.timeit(_vmap_batched_apply_matrix_, number=loops)
print("{} ms per loop".format(t/loops*1000))

