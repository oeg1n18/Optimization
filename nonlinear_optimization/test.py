from gradient_descent import gradient_descent
import numpy as np
from jax import jit, jacfwd, jacrev, grad
import jax.numpy as jnp


def fun(x):
    return 3*x[0]**2 + 3

x0 = jnp.array([1.0]);

grad_descent = gradient_descent('grad_descent', fun, x0, 30)
grad_descent.newtons()
print(grad_descent.x)