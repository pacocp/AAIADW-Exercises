import jax.numpy as jnp
from jax import random
from jax import vmap

def f(x):
    return jnp.sin(x[:,0])*jnp.cos(x[:,1])+jnp.sin(0.5*x[:,0])*jnp.cos(0.5*x[:,1])

def gradient(x):
  grad = jnp.zeros_like(x)
  grad = grad.at[0].set(jnp.cos(x[0])*jnp.cos(x[1]) + 0.5*jnp.cos(0.5*x[0])*jnp.cos(0.5*x[1]))
  grad = grad.at[1].set(-jnp.sin(x[0])*jnp.sin(x[1]) - 0.5*jnp.sin(0.5*x[0])*jnp.sin(0.5*x[1]))
  return grad

def gradient_descent(x0, lr=0.1, n_iters=100):
    points = [x0]
    for i in range(n_iters):
        x0 = x0 - lr * gradient(x0)
        points.append(x0)
    return points

if __name__ == '__main__':
    # testing vmap gradient
    key = random.key(1701)
    x = random.randint(key, minval=1, maxval=10, shape=(5,2))
    print(x)
    gradient_vmap = vmap(gradient, in_axes=0)
    gradient_result = gradient_vmap(x)
    print(gradient_result)

    start_point = jnp.array([4,4])
    points = gradient_descent(start_point, n_iters=5)
    print(points)
