from net import Net
import numpy as np 
import torch 
from torch.autograd.functional import hessian
from torch import autograd


class Problem():
    def __init__(self, func, m: int, n: int):
        self.model = Net(func, m, n)
        self.m = m 
        self.n = n 

    def x_encode(self, x, requires_grad=False):
        inputs = np.empty((self.m, self.n))
        x = x.reshape(self.m-2, self.n-2)
        inputs[1:-1, 1:-1] = x
        self.model.bound_constrain(inputs)
        inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=requires_grad)
        return inputs

    def obj(self, x):
        x = self.x_encode(x)
        loss = self.model.obj(x)
        return loss.item()

    def grad(self, x):
        x = self.x_encode(x, requires_grad=True)
        grad_x = autograd.grad(self.model.obj(x), x)[0].numpy()
        grad_x = grad_x[1:-1, 1:-1]
        return grad_x.flatten()

    def hess(self, x):
        x = self.x_encode(x)
        hess_x = hessian(self.model.obj, x).numpy()
        hess_x = hess_x[1:-1, 1:-1, 1:-1, 1:-1]
        length = (self.m-2) * (self.n-2)
        hess_x = hess_x.reshape(length, length)
        assert np.linalg.norm(hess_x-hess_x.T) < 1e-5, np.linalg.norm(hess_x)
        
        return hess_x

if __name__ == "__main__":
    m, n = 3, 3
    from boundary import FuncClass
    func_class = FuncClass()
    problem = Problem(func_class.func1, m, n)
    # x = np.random.randn((m-2)*(n-2))
    x = np.ones((m-2)*(n-2))
    print(problem.hess(x))




        