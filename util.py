"""
Utility functions
"""
import numpy as np
import params as hp
from numpy.fft import fft2, ifft2

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
    return v + h


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
    grad = get_gradient(img)
    if np.linalg.norm(grad) ** 2 >= hp.lmda / miu:
        return grad
    else:
        return np.zeros_like(grad)
    

def get_g_gradient(g, dv, dh):
    """
    compute the vertical and horizontal gradient of g
    first break up g into two components
    """
    if np.array_equal(g, np.zeros_like(g)):
        return dv, dh
    else:
        return np.zeros_like(dv), np.zeros_like(dh)
    

def pad_kernel(kernel, output_size):
    """
    pad the current kernel to the output size
    """
    assert len(kernel.shape) == 2
    padded_kernel = np.pad(kernel, pad_width=((0, output_size[0] - kernel.shape[0]), (0, output_size[1] - kernel.shape[1])), mode='constant')  # pad the kernel
    # padded_kernel = np.repeat(padded_kernel[:,:,None], output_size[2], axis=2)  # add channel dimension for kernel

    assert padded_kernel.shape == output_size, (kernel.shape, output_size)
    return padded_kernel


def get_latent(u, g, latent_img, blur_img, kernel, beta, miu):
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
    grad = get_gradient(latent_img)
    dv, dh, *_ = np.gradient(latent_img)
    gv, gh = get_g_gradient(g, dv, dh)
    fg = np.conj(fft2(dh)) * fft2(gh) + np.conj(fft2(dv)) * fft2(gv)
    img_kernel = pad_kernel(kernel, blur_img.shape)
    a = np.conj(fft2(img_kernel)) * fft2(blur_img) + beta * fft2(u) + miu * fg
    grad_kernel = pad_kernel(kernel, grad.shape)
    b = np.conj(fft2(grad_kernel)) * fft2(grad_kernel) + beta + miu * np.conj(fft2(grad)) * fft2(grad)
    return ifft2(a / b)
