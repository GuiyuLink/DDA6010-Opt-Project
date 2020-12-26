import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian

# from boundary import func1

class Net(nn.Module):
    def __init__(self, func, m: int, n: int):
        super(Net, self).__init__()
        self.m = m
        self.n = n 
        self.func = func 
        self.delta1 = 1 / (m-1) 
        self.delta2 = 1 / (n-1)
        # initialize
        self.x = torch.zeros((self.m, self.n), dtype=torch.float32)
        # self.x = torch.randn((self.m, self.n), dtype=torch.float32)
        # boundary in place
        self._bound_constrain()
        self.x = nn.Parameter(self.x)
        # self.x.requires_grad = True

    def _bound_constrain(self):
        # (0,0) -> (0,1) & (1,0) -> (1,1)
        for i in range(self.n):
            self.x[0,i] = self.func(0, i*self.delta2)
            self.x[-1,i] = self.func(1, i*self.delta2)
        # (0,0) -> (1,0) & (0,1) -> (1,1)
        for i in range(self.m):
            self.x[i,0] = self.func(i*self.delta1, 0)
            self.x[i,-1] = self.func(i*self.delta1, 1)

    def bound_constrain(self, x):
        # (0,0) -> (0,1) & (1,0) -> (1,1)
        for i in range(self.n):
            x[0,i] = self.func(0, i*self.delta2)
            x[-1,i] = self.func(1, i*self.delta2)
        # (0,0) -> (1,0) & (0,1) -> (1,1)
        for i in range(self.m):
            x[i,0] = self.func(i*self.delta1, 0)
            x[i,-1] = self.func(i*self.delta1, 1)

    def forward(self):
        return self.obj(self.x)

    def obj(self, x):
        assert x.shape == (self.m, self.n)
        # 下三角形
        axis_D1 = self.delta2 * (x[:-1,:-1] - x[1:,:-1])
        axis_D2 = self.delta1 * (x[1:,1:] - x[1:, :-1])
        axis_D3 = torch.ones(axis_D1.shape) * self.delta1 * self.delta2 
        S_D = torch.sqrt((axis_D1.pow(2) + axis_D2.pow(2) + axis_D3.pow(2))).sum() / 2
        # 上三角形
        axis_L1 = self.delta2 * (x[:-1,1:] - x[1:,1:])
        axis_L2 = self.delta1 * (x[:-1,1:] - x[:-1, :-1])
        axis_L3 = torch.ones(axis_L1.shape) * self.delta1 * self.delta2
        S_L = torch.sqrt((axis_L1.pow(2) + axis_L2.pow(2) + axis_L3.pow(2))).sum() / 2
        return S_L + S_D

    def obj_vec(self, x):
        x = x.reshape(self.m, self.n)
        return self.obj(x)



if __name__ == "__main__":
    from boundary import FuncClass
    func_class = FuncClass()
    model = Net(func_class.func1, 3, 3)
    loss = model()
    loss.backward()
    print(model.x)
    print(model.x.grad)
    model.zero_grad()
    print(model.x.grad)
    print(loss)
    

    for i in range(10):
        learning_rate = 0.1
        model.zero_grad()
        loss = model()
        loss.backward()
        for f in model.parameters():
            f.data[1,1].sub_(f.grad.data[1,1] * learning_rate)
        print(model.x)

    # print(hessian(model.obj_vec, model.x.flatten()))

