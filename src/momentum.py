#!/usr/bin/env python3

import numpy as np
import time

def Momentum_method(obj, grad, x, tol, beta,sigma):
    """
    obj: objective function from R^n to R
    grad: gradient of the objection, from R^n to R^n
    x: initial point, np.array([x1, x2, ..., xn])
    tol: stoppting criterion, the algorithm stop when the norm of gradient <= tol
    beta: fixed beta for extrapolitation
    sigma: multiplier for searching Lipchitz constant

    """
    iterations = 0
    points = [x]
    grad_norm = [np.linalg.norm(grad(x))]
    objs = [obj(x)]
    start_time = time.time()
    times = [0]
    difference=0
    while grad_norm[-1] > tol:
        # initial step size
        grad_x = grad(x)
        #calculate unknown Lipchitz-const
        L_k=1
        if iterations>0:
            while objs[-1]>objs[-2]+grad(points[-2]).dot(difference)+0.5*L_k*np.linalg.norm(difference)**2:
                L_k=L_k/sigma
        alpha=2*(1-beta)/L_k
        # update point
        y=x+beta * difference
        x = y - alpha * grad_x
        # print("dwt | x: ", x)
        # print("dwt | a: ", a)
        # print("x: ", x, " | value: ", obj(x))
        # print("--------------------------------")
        points.append(x)
        # update iterations
        difference=points[-1]-points[-2]
        iterations += 1
        grad_norm.append(np.linalg.norm(grad_x))
        print('iter: {}, obj: {:.10f}, grad_norm: {:.6g}'.format(iterations, obj(x), grad_norm[-1]))
        times.append(time.time()-start_time)
        objs.append(obj(x))

    return points, objs, grad_norm, times


def Nestrov_acc(obj, grad, x, tol, alpha):
    """
    obj: objective function from R^n to R
    grad: gradient of the objection, from R^n to R^n
    x: initial point, np.array([x1, x2, ..., xn])
    tol: stoppting criterion, the algorithm stop when the norm of gradient <= tol
    alpha: fixed step length

    """
    iterations = 0
    points = [x]
    grad_norm = [np.linalg.norm(grad(x))]
    objs = [obj(x)]
    start_time = time.time()
    times = [0]
    difference=0
    theta_old=1
    while grad_norm[-1] > tol:
        # initial step size
        grad_x = grad(x)
        #calculate unknown Lipchitz-const
        theta_new=2/(iterations+3)
        beta=theta_new/theta_old-theta_new
        # update point
        y=x+beta * difference
        x = y - alpha * grad(y)
        # print("dwt | x: ", x)
        # print("dwt | a: ", a)
        # print("x: ", x, " | value: ", obj(x))
        # print("--------------------------------")
        points.append(x)
        # update iterations
        difference=points[-1]-points[-2]
        theta_old=theta_new
        iterations += 1
        grad_norm.append(np.linalg.norm(grad_x))
        print('iter: {}, obj: {:.10f}, grad_norm: {:.6g}'.format(iterations, obj(x), grad_norm[-1]))
        times.append(time.time()-start_time)
        objs.append(obj(x))

    return points, objs, grad_norm, times