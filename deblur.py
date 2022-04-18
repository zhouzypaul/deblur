"""
This file handles the debluring of images
this is where the magic comes together
"""
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from get_data import parse_dataset


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
    # TODO: Paul
    pass


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


def main():
    """
    interleave kernel and latent image estimation
    """
    # TODO: Paul
    pass


if __name__ == "__main__":
    main()
