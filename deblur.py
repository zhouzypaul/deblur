"""
This file handles the debluring of images
this is where the magic comes together
"""
import numpy as np
from numpy.fft import fft2, ifft2
import sklearn
import matplotlib.pyplot as plt
import util
import params as hp

from get_data import parse_dataset


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
    beta = 2 * hp.lmda * hp.sigma
    # repeat 
    while beta <= hp.beta_max:
        # solve for u using (10)
        u = util.get_u(latent_img)
        miu = 2 * hp.lmda

        # repeat
        while miu <= hp.miu_max:
            # solve for g using (11)
            g = util.get_g(latent_img, miu)
            # solve for x using (8)
            latent_img = util.get_latent(u, g, latent_img, blur_img, kernel, beta, miu)
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
