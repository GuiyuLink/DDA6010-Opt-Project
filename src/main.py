import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from BB import BB_gradient, gradient_method_backtracking, BB_gradient_nonmonotone
from boundary import FuncClass
from newton import globalized_newton
from problem_wrap import Problem


def plot3D(m, n, Z):
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# method: 0: basic GD with backtracking, 1: globalized Newton method,  2: L-BFGS
# 3: nonmonotone line search procedures, 4: Barzilai-Borwein steps 5: inertial techniques and momentum
# 6: compact representation of the L-BFGS update
METHOD = 4
func_num = 0
FUNC = 'func_class.func{}'.format(func_num)
OPT = [2.0725090145, 2.1410255432128906, 1.2063673734664917, 1.0187329445407338, 3.086641232606508][func_num]

if __name__ == "__main__":
    # parameters
    m, n = 50, 50
    tol = 1e-5
    s, sigma, gamma = 1, 0.5, 0.1
    beta1 = beta2 = 1e-6
    p = 0.1

    func_class = FuncClass()
    problem = Problem(eval(FUNC), m, n)

    # initial
    # x = np.random.randn((m-2)*(n-2))
    x = np.zeros((m-2)*(n-2))

    # algorithm
    if METHOD == 0:
        points, objs, grad_norm, times = gradient_method_backtracking(problem.obj, problem.grad, x, tol, s, sigma, gamma)
    elif METHOD == 1:
        points, objs, grad_norm, times = globalized_newton(problem.obj, problem.grad, problem.hess, x,  tol, s, sigma, gamma, beta1, beta2, p)
    elif METHOD == 2:
        pass 
    elif METHOD == 3:
        # points, objs, grad_norm, times = BB_gradient_nonmonotone(problem.obj, problem.grad, x,  tol, s, sigma)
        pass
    elif METHOD == 4:
        points, objs, grad_norm, times = BB_gradient(problem.obj, problem.grad, x, tol, s, sigma, gamma)
    elif METHOD == 5:
        pass 
    elif METHOD == 6:
        pass 

    # plot gap
    obj = np.array(objs)
    gap = np.abs(obj-OPT) / max(1, OPT)
    plt.figure()
    plt.plot(gap)
    plt.xlabel('iterations')
    plt.ylabel('gap')
    plt.show()

    plt.figure()
    plt.plot(times, gap)
    plt.xlabel('cpu time (second)')
    plt.ylabel('gap')
    plt.show()

    # plot 
    plt.figure()
    plt.plot(grad_norm)
    plt.xlabel('iterations')
    plt.ylabel('grad_norm')
    plt.yscale('log') 
    plt.show()

    plt.figure()
    plt.plot(times, grad_norm)
    plt.xlabel('cpu time (second)')
    plt.ylabel('grad_norm')
    plt.yscale('log') 
    plt.show()

    # plot
    inputs = np.empty((m, n))
    x = points[-1]
    x = x.reshape(m-2, n-2)
    inputs[1:-1, 1:-1] = x
    problem.model.bound_constrain(inputs)
    plot3D(m, n, inputs.T)
    plt.savefig('../report/{}_method{}.pdf'.format(FUNC, METHOD))
    plt.show()
