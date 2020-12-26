#!/usr/bin/env python3

import numpy as np

def gradient_method_backtracking(obj, grad, x, tol, s, sigma, gamma):
    """
    obj: objective function from R^n to R
    grad: gradient of the objection, from R^n to R^n
    x: initial point, np.array([x1, x2, ..., xn])
    tol: stoppting criterion, the algorithm stop when the norm of gradient <= tol
    s: the initial value for step size
    sigma: the step size shold by multipled by sigma when Armijo condition not satisified
    gamma: parameters of Armijo condition
    """
    iterations = 0
    points = [x]
    while np.linalg.norm(grad(x)) > tol:
        # initial step size
        a = s
        # store the gradient of x^k
        grad_x = grad(x)
        # estimate Armijo condition
        while obj(x - a * grad_x) - obj(x) > - gamma * a * grad_x.dot(grad_x):
            a = a * sigma

        # update point
        x = x - a * grad_x
        # print("dwt | x: ", x)
        # print("dwt | a: ", a)
        # print("x: ", x, " | value: ", obj(x))
        # print("--------------------------------")
        points.append(x)
        # update iterations
        iterations += 1
        print(iterations, obj(x), np.linalg.norm(grad_x))

    return points, iterations

def BB_gradient(obj, grad, x, tol, s, sigma, gamma, initial_method="backtracking"):
    """
    Solve problem by Barzilai-Borwein gradient method.
    x_{k+1} = x_{k} - a_k g_k

    obj: objective function from R^n to R
    grad: gradient of the objection, from R^n to R^n
    x: initial point, np.array([x1, x2, ..., xn])
    tol: stoppting criterion, the algorithm stop when the norm of gradient <= tol
    s: the initial value for step size
    sigma: the step size shold by multipled by sigma when Armijo condition not satisified
    gamma: parameters of Armijo condition
    """
    # --- first update by initial _linesearch_method ---
    print("-"*30)
    print("BB gradient start")
    iterations = 0
    points = [x]
    last_grad_x = grad(x)
    # initial step size
    a = s
    # estimate Armijo condition
    while obj(x - a * last_grad_x) - obj(x) > - gamma * a * last_grad_x.dot(last_grad_x):
        a = a * sigma
    # update point
    x = x - a * last_grad_x
    points.append(x)
    iterations += 1
    current_grad_x = grad(x)
    # --- iteration by BB steps ---
    while np.linalg.norm(current_grad_x) > tol:
        # compute step size a
        z = x - points[-2] # z = x_k - x_{k-1}
        y = current_grad_x - last_grad_x # y = g_k - g_{k-1}
        a = z.dot(y) / y.dot(y)
        x = x - a * current_grad_x
        points.append(x)
        iterations += 1
        last_grad_x = current_grad_x
        current_grad_x = grad(x)
        print(iterations, obj(x), np.linalg.norm(current_grad_x))

        # print("iterations: ", iterations, " | x: ", x, " | grad: ", current_grad_x, " | norm: ", np.linalg.norm(current_grad_x))

    print("iterations: ", iterations, " | x: ", x, " | grad: ", current_grad_x, " | norm: ", np.linalg.norm(current_grad_x))
    print("BB gradient end")
    print("-"*30)
    return points, iterations

def BB_gradient_nonmonotone(obj, grad, x, tol, s, sigma, gamma, sigma_1, sigma_2, a_m, a_M, M, delta, rho):
    """
    Solve problem by Barzilai-Borwein gradient method.
    x_{k+1} = x_{k} - a_k g_k

    obj: objective function from R^n to R
    grad: gradient of the objection, from R^n to R^n
    x: initial point, np.array([x1, x2, ..., xn])
    tol: stoppting criterion, the algorithm stop when the norm of gradient <= tol
    s: the initial value for step size
    sigma: the step size shold by multipled by sigma when Armijo condition not satisified
    gamma: parameters of Armijo condition
    sigma_1, sigam_2: parameter to choose lambda in each iteration
    a_m, a_M, delta: choosen part of a, if not set it to delta
    M: length of memoty
    rho: paramter for nonmonotone linesearch
    """
    # --- first update by initial _linesearch_method ---
    print("-"*30)
    print("BB gradient nonmonotone start")
    iterations = 0
    points = [x]
    objs = [obj(x)]
    grad_x = grad(x)
    # initial step size
    a = s
    # estimate Armijo condition
    while obj(x - a * grad_x) - obj(x) > - gamma * a * grad_x.dot(grad_x):
        a = a * sigma
    # update point
    x = x - a * grad_x
    points.append(x)
    objs.append(obj(x))
    iterations += 1
    current_grad_x = grad(x)
    while np.linalg.norm(current_grad_x) > tol:
        last_grad_x = grad(x)
        if (a_m >= a) or (a >= a_M):
            a = delta
        lam = 1.0/a
        max_prev = np.max(objs[-M:])
        # set lam
        while obj(x - lam * current_grad_x) > max_prev - rho * lam * current_grad_x.dot(current_grad_x):
            # TODO choose sigma in [sigma_1, sigma_2]
            lam = lam * (sigma_1 + sigma_2) / 2
        x = x - lam * current_grad_x
        points.append(x)
        objs.append(obj(x))
        iterations += 1
        current_grad_x = grad(x)
        a = - current_grad_x.dot(current_grad_x - last_grad_x) / (lam * current_grad_x.dot(current_grad_x))
        # print("dwt | a: ", a)
        print(iterations, obj(x), np.linalg.norm(grad_x))
        # print("iterations: ", iterations, " | x: ", x, " | grad: ", current_grad_x, " | norm: ", np.linalg.norm(current_grad_x))
    print("iterations: ", iterations, " | x: ", x, " | grad: ", current_grad_x, " | norm: ", np.linalg.norm(current_grad_x))
    print("BB gradient nonmonotone end")
    print("-"*30)
    return points, iterations
