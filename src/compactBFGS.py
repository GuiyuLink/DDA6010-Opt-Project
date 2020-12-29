import numpy as np
import time


def BFGS_DFP(obj, grad, hess, x, tol, s, sigma, gamma, beta_1, beta_2, p, lam):
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
    print("-" * 30)
    print("compact_BFGS start")
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
    times = [time.time() - start_time]
    while np.linalg.norm(grad_x) > tol:
        # initial step size
        d = - B_inv.dot(grad_x)
        norm_d = np.linalg.norm(d)
        # if - d.dot(grad_x) < np.min([beta_1, beta_2 * (norm_d ** p)]) * norm_d * norm_d:
        #    d = - grad_x
        #    print('xxxx')
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
        B_inv_BFGS = B_inv + ((s_k - Hy).dot(s_k.T) + s_k.dot((s_k - Hy).T)) / ((s_k.T).dot(y_k)) - ((s_k - Hy).T).dot(
            y_k) / ((s_k.T).dot(y_k)) ** 2 * s_k.dot(s_k.T)
        B_inv_DFP = B_inv + (s_k).dot(s_k.T) / (s_k.T).dot(y_k) - (Hy).dot(Hy.T) / (y_k.T).dot(Hy)
        B_inv = lam * B_inv_DFP + (1 - lam) * B_inv_BFGS
        # update iterations
        iterations += 1
        grad_x = grad(x)
        grad_norm.append(np.linalg.norm(grad_x))
        objs.append(obj(x))
        print('iter: {}, obj: {:.10f}, grad_norm: {:.6g}'.format(iterations, objs[-1], grad_norm[-1]))
        times.append(time.time() - start_time)

    print("iterations: ", iterations, " | x: ", x, " | grad: ", grad_x, " | norm: ", np.linalg.norm(grad_x))
    print("compact_BFGS end")
    print("-" * 30)
    return points, objs, grad_norm, times


def L_BFGS(obj, grad, hess, x, tol, s, sigma, gamma, m):
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
    print("-" * 30)
    print("L_BFGS start")
    start_time = time.time()
    iterations = 0
    points = [x]
    grad_x = grad(x)
    grad_norm = [np.linalg.norm(grad_x)]
    objs = [obj(x)]
    s_ks = []
    y_ks = []
    rho_ks = []
    # set the initial B as hessian
    # B = hess(x)
    # B_inv = np.linalg.inv(B)
    times = [time.time() - start_time]
    while np.linalg.norm(grad_x) > tol:
        # initial step size
        total_length = min(m, iterations)
        if iterations < 1:
            d = -grad(x)
        else:
            # double update loop
            q = grad_x
            a_s = []
            for i in range(1, total_length + 1):
                a = rho_ks[-i] * (s_ks[-i]).T.dot(q)
                q = q - a * y_ks[-i]
                a_s.append(a)
            gam = (s_ks[-1].T.dot(s_ks[-1])) / (s_ks[-1].T.dot(y_ks[-1]))
            r = gam * q
            for i in range(total_length, 0, -1):
                b = rho_ks[-i] * y_ks[-i].T.dot(r)
                r = r + (a_s[::-1][-i] - b) * s_ks[-i]
            d = -r
        # Armijo
        a = s
        while obj(x + a * d) - obj(x) > gamma * a * d.dot(grad_x):
            a = a * sigma
        # update point
        x = x + a * d
        points.append(x)
        # Calculate B_k
        s_k = (a * d)[:, np.newaxis]
        y_k = (grad(x) - grad_x)[:, np.newaxis]
        rho_k = np.squeeze((1 / (s_k.T).dot(y_k)))
        s_ks.append(np.squeeze(s_k))
        y_ks.append(np.squeeze(y_k))
        rho_ks.append(rho_k)
        # update iterations
        iterations += 1
        grad_x = grad(x)
        grad_norm.append(np.linalg.norm(grad_x))
        objs.append(obj(x))
        print('iter: {}, obj: {:.10f}, grad_norm: {:.6g}'.format(iterations, objs[-1], grad_norm[-1]))
        times.append(time.time() - start_time)

    print("iterations: ", iterations, " | x: ", x, " | grad: ", grad_x, " | norm: ", np.linalg.norm(grad_x))
    print("L_BFGS end")
    print("-" * 30)
    return points, objs, grad_norm, times
