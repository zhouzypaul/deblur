import numpy as np

"""
Conjugrate gradient method iterative implementation
"""


def conjugate_gradient(x, b, max_it, tol, ax_func, func_param):
    '''
    Solves a system of linear equations using the conjugate gradient method.
    args:
        x: initial estimate of x
        b: known b from Ax = b
        max_it: maximum number of iterations
        tol: residual tolerance
        ax_func: calculates Ap, multiplication between A and conjugate basis vectors
        fund_param: parameters to ax_fund
    return:
        x which solves Ax = b
    '''

    # residual
    r = b - ax_func(x, func_param)

    # initialize first basis vector
    p = r

    rsold = np.sum(r ** 2)

    for _ in range(max_it):
        # define alpha for iteration k
        ap = ax_func(p, func_param)
        alpha = rsold / np.sum(p * ap)

        # update solution x
        x += alpha * p

        # update residual
        r -= alpha * ap
        rsnew = np.sum(r ** 2)

        # stop when residual is sufficiently small
        if np.sqrt(rsnew) < tol:
            break

        # update conjugate vector estimates
        p = r + rsnew / rsold * p

        rsold = rsnew

    return x
