from problem_wrap import Problem
import numpy as np 
from boundary import FuncClass
from newton import globalized_newton
from BB import gradient_method_backtracking, BB_gradient

def plot3D(m, n, Z):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, m)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()

if __name__ == "__main__":
    m, n = 80, 80
    tol = 1e-5
    s, sigma, gamma = 1, 0.5, 0.1
    beta1 = beta2 = 1e-6
    p = 0.1
    func_class = FuncClass()
    problem = Problem(func_class.func5, m, n)
    # x = np.random.randn((m-2)*(n-2))
    x = np.zeros((m-2)*(n-2))
    points, iterations = (globalized_newton(problem.obj, problem.grad, problem.hess, x,  tol, s, sigma, gamma, beta1, beta2, p))
    # print(gradient_method_backtracking(problem.obj, problem.grad, x, tol, s, sigma, gamma))
    # points, iterations = (BB_gradient(problem.obj, problem.grad, x, tol, s, sigma, gamma))

    # plot
    inputs = np.empty((m, n))
    x = points[-1]
    x = x.reshape(m-2, n-2)
    inputs[1:-1, 1:-1] = x
    problem.model.bound_constrain(inputs)
    plot3D(m,n,inputs)
