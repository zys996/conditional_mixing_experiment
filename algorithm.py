import numpy as np

class ULA_optimizer:
    def __init__(self, eps, d):
        """
        eps: step size
        d: data dimension
        """
        self.eps = eps
        self.d = d
        if d > 1:
            self.mean = np.zeros(d)
            self.cov = np.identity(d)
        elif d == 1:
            self.mean = 0
            self.cov = 1

    def update(self, x_t, grad_x):
        if self.d > 1:
            return x_t - self.eps * grad_x + np.sqrt(2 * self.eps)* np.random.multivariate_normal(self.mean, self.cov, size = len(x_t))
        elif self.d == 1:
            return x_t - self.eps * grad_x + np.sqrt(2 * self.eps)* np.random.normal(self.mean, self.cov, size = len(x_t))

