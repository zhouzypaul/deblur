"""
This file handles the debluring of images
this is where the magic comes together
"""
import numpy as np
from numpy.fft import fft2, ifft2
import sklearn
import matplotlib.pyplot as plt

from get_data import parse_dataset


# hyperparameters from the paper
lmda = 4e-3
gamma = 2
sigma = 1
beta_max = 2**3
miu_max = 1e5


def get_gradient(img):
    """
    this function calculates the gradient of the image
    args:
        img: an image
    returnn:
        the gradient of the image
    """
    # TODO: Ed
    pass


def estimating_latent_image_with_blur_kernel(blur_img, kernel):
    """
    this function estimates the latent clean image using the kernel as input
    this is algorithm 1 and section 3.1 in the paper
    args:
        blur_img: the blur image
        kernel: the blur kernel
    return:
        intermediate latente image
    """
    latent_img = blur_img
    beta = 2 * lmda * sigma
    # repeat 
    while beta <= beta_max:
        # solve for u using (10)
        u = latent_img if np.linalg.norm(latent_img)**2 >= lmda * sigma / beta else np.zeros_like(latent_img)
        miu = 2 * lmda

        # repeat
        while miu <= miu_max:
            # solve for g using (11)
            grad = get_gradient(latent_img)
            g = grad if np.linalg.norm(grad)**2 >= lmda/miu else np.zeros_like(grad)
            # solve for x using (8)
            F_G = np.conj(fft2(np.gradient(latent_img, axis=1))) * fft2(np.gradient(g, axis=1)) \
                + np.conj(fft2(np.gradient(latent_img, axis=0))) * fft2(np.gradient(g, axis=0))
            numerator = np.conj(fft2(kernel)) * fft2(blur_img) + beta * fft2(u) + miu * F_G
            denominator = np.conj(fft2(kernel)) * fft2(kernel) + beta + miu * np.conj(fft2(grad)) * fft2(grad)
            latent_img = ifft2(numerator / denominator)
            miu = 2 * miu
    return latent_img


def estimating_blur_kernel(blur_img):
    """
    this function estimate the blur kernel
    this is algorithm 2 and section 3.2 in the paper
    args:
        blur_img: the blur image
    return:
        blur kernel and intermediate latent image
    """
    # TODO: Alan
    pass


def remove_artifact(blur_img):
    """
    this removes the artifacts as detailed in section 3.3 of the paper
    """
    # TODO: Luca
    pass


def main(data_path='data/original_images/'):
    """
    interleave kernel and latent image estimation
    load the original images and deblur them, and plot them together
    """
    blurred_images = parse_dataset(data_path)  # there should be 15 of them
    for img in blurred_images:
        blur_kernel, restored_img = estimating_blur_kernel(img)
        plt.imshow(restored_img)
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
