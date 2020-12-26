import torch
from torch.autograd.functional import hessian

def pow_reducer(x):
    return x.pow(3).sum() + torch.norm(x)

inputs = torch.rand(2, 3)#.flatten()
# print(torch.randn(2, 3, dtype=torch.float32))
print(hessian(pow_reducer, inputs))
y = hessian(pow_reducer, inputs)
x = hessian(pow_reducer, inputs).reshape(6,6)
print(x)
print(torch.norm(x-x.T))