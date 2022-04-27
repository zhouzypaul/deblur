"""
this file handles the evaluation of the debluring methods
"""
from cv2 import blur
from matplotlib import pyplot as plt
from get_data import parse_dataset
import numpy as np


def get_average_psnr(img):
    """
    this function calculates the average psnr of an image
    args:
        img: a list of images produced by different blurs on the same image
    return:
        the average psnr
    """
    # TODO: Alan
    return 10


def main(data_path='data/original_images/', kernel_path='data/kernels/'):
    """
    plot the results of average psnr
    """
    truth_images, blurred_images = parse_dataset(data_path, kernel_path)
    psnr = []
    labels = []

    for i, img in enumerate(blurred_images):
        psnr.append(get_average_psnr(img))
        labels.append(str(i + 1))

    plt.bar(labels, psnr)
    plt.show()


if __name__ == "__main__":
    main()
