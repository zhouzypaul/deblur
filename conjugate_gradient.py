import numpy as np

def conjugate_gradient(x, b, max_it, tol, ax_func, func_param):
    r = b - ax_func(x, func_param)
    p = r
    rsold = np.sum(r ** 2)

    for _ in range(max_it):
        ap = ax_func(p, func_param)
        alpha = rsold / np.sum(p * ap)
        x += alpha * p
        r -= alpha * ap
        rsnew = np.sum(r ** 2)

        if np.sqrt(rsnew) < tol:
            break

        p = r + rsnew / rsold * p
        rsold = rsnew


