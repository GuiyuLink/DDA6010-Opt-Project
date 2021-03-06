import numpy as np
import time

def projected_gradient(obj, grad, x, tol, s, sigma, gamma, lambda_k, lower_bound):
    """
    obj: objective function from R^n to R
    grad: gradient of the objection, from R^n to R^n
    x: initial point, np.array([x1, x2, ..., xn])
    tol: stoppting criterion, the algorithm stop when the norm of gradient <= tol
    s: the initial value for step size
    sigma: the step size shold by multipled by sigma when Armijo condition not satisified
    gamma: parameters of Armijo condition
    lambda_k: weight in direction, should be bounded
    lower_bound: lower bound vector
    """
    iterations = 0
    # Project the initial point on the box set
    start_time = time.time()
    x = np.maximum(x, lower_bound)
    points = [x]
    d_norm = tol + 1
    grad_norm = [d_norm]
    objs = [obj(x)]
    times = [time.time()-start_time]
    while d_norm > tol:
        # initial step size
        a = s
        # store the gradient of x^k
        grad_x = grad(x)
        # calculate the projected direction
        d = np.maximum(x-lambda_k(iterations)*grad_x, lower_bound) - x
        # estimate Armijo condition
        while obj(x + a * d) - obj(x) > gamma * a * grad_x.dot(d):
            a = a * sigma

        # update point
        x = x + a * d
        points.append(x)
        # update iterations
        iterations += 1
        d_norm = np.linalg.norm(d)
        grad_norm.append(d_norm)
        objs.append(obj(x))
        print('iter: {}, obj: {:.10f}, grad_norm: {:.6g}'.format(iterations, objs[-1], grad_norm[-1]))
        times.append(time.time()-start_time)

    return points, objs, grad_norm, times