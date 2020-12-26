import numpy as np

class FuncClass():
    def func0(self, x, y):
        return 1 + np.sin(2*np.pi*x)

    def func1(self, x, y):
        return 1 + np.cos(1/(x + 0.001))

    def func2(self, x, y):
        return 1/2 - np.abs(y - 1/2)

    def func3(self, x, y):
        return 1/(1+np.exp(x*y))

    def func4(self, x, y):
        return 1 + np.arcsin(-1+2*np.sqrt(x*y))