import numpy as np
import params as hp
from numpy.fft import fft2, ifft2

def get_gradient(img):
    v, h = np.gradient(img)
    return np.hstack((h, v)).T


def get_u(img):
    if np.linalg.norm(img) ** 2 >= hp.lmda * hp.sigma / hp.beta:
        return img
    else:
        return np.zeros_like(img)


def get_g(img, miu):
    grad = get_gradient(img)
    if np.linalg.norm(grad) ** 2 >= hp.lmda / miu:
        return grad
    else:
        return np.zeros_like(grad)


def get_latent(u, g, latent_img, blur_img, kernel, beta, miu):
    grad = get_gradient(latent_img)
    dv, dh = np.gradient(latent_img)
    gv, gh = np.gradient(g)
    fg = np.conj(fft2(dh)) * fft2(gh) + np.conj(fft2(dv)) * fft2(gv)
    a = np.conj(fft2(kernel)) * fft2(blur_img) + beta * fft2(u) + miu * fg
    b = np.conj(fft2(kernel)) * fft2(kernel) + beta + miu * np.conj(fft2(grad)) * fft2(grad)
    return ifft2(a / b)
