"""
Utility functions
"""
import numpy as np
import params as hp
from numpy.fft import fft2, ifft2

SOBEL_V = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_H = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def get_gradient(img):
    """
    Computes the gradient of an image
    args:
        img: the image
    return:
        the gradient of the image g = (g_h, g_v)^T where g_h and g_v are the 
        horizontal and vertical gradients, respectively
    """
    v, h, *_ = np.gradient(img)
    return np.hstack((h, v)).T


def get_u(img, beta):
    """
    Computes u according to equation 10
    args:
        img: the image
    return:
        the auxiliary variable u
    """
    if np.linalg.norm(img) ** 2 >= hp.lmda * hp.sigma / beta:
        return img
    else:
        return np.zeros_like(img)


def get_g(img, miu):
    """
    Computes g according to equation 11
    args:
        img: the image
        miu: hyperparameter
    return:
        the auxiliary variable g
    """
    v, h, *_ = np.gradient(img)
    if np.linalg.norm(np.hstack((h, v))) ** 2 >= hp.lmda / miu:
        return v, h
    else:
        return np.zeros_like(v), np.zeros_like(h)
    

def pad_kernel(kernel, output_size):
    """
    pad the current kernel to the output size
    """

    padded_kernel = np.pad(kernel, pad_width=((0, output_size[0] - kernel.shape[0]), (0, output_size[1] - kernel.shape[1])), mode='constant')  # pad the kernel
    # padded_kernel = np.repeat(padded_kernel[:,:,None], output_size[2], axis=2)  # add channel dimension for kernel
    return padded_kernel


def get_latent(u, g, blur_img, kernel, beta, miu):
    """
    Computes x according to equation 8
    args:
        u: auxiliary variable
        g: auxiliary variable
        latent_img: deblured image
        blur_img: blurred image
        kernel: estimated blur kernel
        beta: hyperparameter
        miu: hyperparameter
    """

    shape = blur_img.shape
    gv, gh = g
    sv, sh = pad_kernel(SOBEL_V, shape), pad_kernel(SOBEL_H, shape)
    nabla = pad_kernel(np.sqrt(SOBEL_H ** 2 + SOBEL_V ** 2), shape)
    fg = np.conj(fft2(sh)) * fft2(gh) + np.conj(fft2(sv)) * fft2(gv)
    img_kernel = pad_kernel(kernel, shape)
    a = np.conj(fft2(img_kernel)) * fft2(blur_img) + beta * fft2(u) + miu * fg
    b = np.conj(fft2(img_kernel)) * fft2(img_kernel) + beta + miu * np.conj(fft2(nabla)) * fft2(nabla)
    latent = np.real(ifft2(a / b))
    return latent / np.max(latent)
