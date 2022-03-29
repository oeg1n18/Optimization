from jax import jit, jacfwd, jacrev
import jax.numpy as jnp


class gradient_descent:
    def __init__(self, type, func, x0, max_steps):
        self.type = type
        self.func = func
        self.jacob = jit(jacrev(func))
        self.hessian = jit(jacfwd(jacrev(func)))
        self.max_iter = max_steps
        self.tol = 0.01
        self.x0 = x0
        self.x = 10

        # Grad Descent
        self.lr = 0.1

        # Heavy Ball Descent
        self.beta = 0.9

    def grad_descent(self):
        iter = 0
        self.x = self.x0
        while jnp.max(self.jacob(self.x)) > self.tol and iter < self.max_iter:
            self.x = self.x - self.lr * self.jacob(self.x)
            iter += 1

    def heavy_ball(self):
        iter = 0
        xold = self.x0
        self.x = self.x0
        while iter < self.max_iter:
            xnew = self.x - self.lr * self.jacob(self.x) + self.beta * (self.x - xold)
            xold = self.x
            self.x = xnew
            iter += 1

    def nesterov(self):
        iter = 0
        xold = self.x0
        self.x = self.x0
        while iter < self.max_iter:
            y = self.x + iter / (iter + 3) * (self.x - xold)
            xnew = y - self.lr * self.jacob(y)
            self.x = xnew
            xold = self.x
            iter += 1

    def newtons(self):
        iter = 0
        self.x = self.x0
        while jnp.max(self.jacob(self.x)) > self.tol and iter < self.max_iter:
            self.x = self.x - jnp.linalg.inv(self.hessian(self.x)) * self.jacob(self.x)
