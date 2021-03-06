#!/usr/bin/env python

import numpy as np
import time 

def globalized_newton(obj, grad, hess, x, tol, s, sigma, gamma, beta_1, beta_2, p):
    """
    obj: objective function from R^n to R
    grad: gradient of the objection, from R^n to R^n
    hess: Hessian matrix of the objective function, from R^n to R^{n \times n}
    x: initial point, np.array([x1, x2, ..., xn])
    tol: stoppting criterion, the algorithm stop when the norm of gradient <= tol
    s: the initial value for step size
    sigma: the step size shold by multipled by sigma when Armijo condition not satisified
    gamma: parameters of Armijo condition
    beta_1, beta_2, p : parameter for choosen newton step of not
    """
    print("-"*30)
    print("Globalized Newton start")
    start_time = time.time()
    iterations = 0
    points = [x]
    grad_x = grad(x)
    grad_norm = [np.linalg.norm(grad_x)]
    objs = [obj(x)]
    times = [time.time()-start_time]
    while grad_norm[-1] > tol:
        # initial step size
        d = - np.linalg.solve(hess(x), grad_x)
        norm_d = np.linalg.norm(d)
        if - d.dot(grad_x) < np.min([beta_1, beta_2 * (norm_d ** p)]) * norm_d * norm_d:
            d = - grad_x
            print('xxxx')
        # estimate Armijo condition
        a = s
        while obj(x + a * d) - obj(x) > gamma * a * d.dot(grad_x):
            a = a * sigma
        # update point
        x = x + a * d
        points.append(x)
        # update iterations
        iterations += 1
        grad_x = grad(x)
        grad_norm.append(np.linalg.norm(grad_x))
        print('iter: {}, obj: {:.10f}, grad_norm: {:.6g}'.format(iterations, obj(x), grad_norm[-1]))
        times.append(time.time()-start_time)
        objs.append(obj(x))

    print("iterations: ", iterations, " | x: ", x, " | grad: ", grad_x, " | norm: ", np.linalg.norm(grad_x))
    print("Globalized Newton end")
    print("-"*30)
    return points, objs, grad_norm, times

def BFGS(obj, grad, hess, x, tol, s, sigma, gamma, beta_1, beta_2, p):
    """
    obj: objective function from R^n to R
    grad: gradient of the objection, from R^n to R^n
    hess: Hessian matrix of the objective function, from R^n to R^{n \times n}
    x: initial point, np.array([x1, x2, ..., xn])
    tol: stoppting criterion, the algorithm stop when the norm of gradient <= tol
    s: the initial value for step size
    sigma: the step size shold by multipled by sigma when Armijo condition not satisified
    gamma: parameters of Armijo condition
    beta_1, beta_2, p : parameter for choosen newton step of not
    """
    print("-"*30)
    print("BFGS start")
    start_time = time.time()
    iterations = 0
    points = [x]
    grad_x = grad(x)
    grad_norm = [np.linalg.norm(grad_x)]
    objs = [obj(x)]
    # set the initial B as hessian
    print('Calculate the Initial Hessian')
    B = hess(x)
    B_inv = np.linalg.inv(B)
    times = [time.time()-start_time]
    while np.linalg.norm(grad_x) > tol:
        # initial step size
        d = - B_inv.dot(grad_x)
        norm_d = np.linalg.norm(d)
        if - d.dot(grad_x) < np.min([beta_1, beta_2 * (norm_d ** p)]) * norm_d * norm_d:
            d = - grad_x
            print('xxxx')
        # estimate Armijo condition
        a = s
        while obj(x + a * d) - obj(x) > gamma * a * d.dot(grad_x):
            a = a * sigma
        # update point
        x = x + a * d
        points.append(x)
        # Calculate B_k
        s_k = (a * d)[:, np.newaxis]
        y_k = (grad(x) - grad_x)[:, np.newaxis]
        Hy = B_inv.dot(y_k)
        B_inv = B_inv + ((s_k-Hy).dot(s_k.T)+s_k.dot((s_k-Hy).T))/((s_k.T).dot(y_k)) - ((s_k-Hy).T).dot(y_k)/((s_k.T).dot(y_k))**2 * s_k.dot(s_k.T)
        # update iterations
        iterations += 1
        grad_x = grad(x)
        grad_norm.append(np.linalg.norm(grad_x))
        objs.append(obj(x))
        print('iter: {}, obj: {:.10f}, grad_norm: {:.6g}'.format(iterations, objs[-1], grad_norm[-1]))
        times.append(time.time()-start_time)

    print("iterations: ", iterations, " | x: ", x, " | grad: ", grad_x, " | norm: ", np.linalg.norm(grad_x))
    print("BFGS end")
    print("-"*30)
    return points, objs, grad_norm, times
