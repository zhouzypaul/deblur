import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from motion_deblur import deblur
from get_data import parse_dataset

"""
this file handles the evaluation of the debluring methods
"""


def get_average_psnr(results, truth):
    """
    this function calculates the average psnr of an image
    args:
        img: a list of images produced by different blurs on the same image
    return:
        the average psnr
    """
    values = []
    for img in results:
        values.append(psnr(img, truth))
    return np.mean(np.asarray(values))


def psnr(result, truth):
    res = convert_to_uint(result)
    tru = convert_to_uint(truth)
    mse = np.mean((tru - res) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(mse))


def convert_to_uint(img):
    return np.uint8(255 * (img - np.min(img)) / (np.max(img) - np.min(img)))


def main():
    """
    plot the results of average psnr
    """
    reported_psnr = [27.5, 23, 26.5, 38]
    
    if not os.listdir('results'):
        image_path = 'data/ieee2016/text-images/gt_images'
        kernel_path = 'data/ieee2016/text-images/kernels'
        truth_images, blurred_images = parse_dataset(image_path, kernel_path)
        results = []
        values = []
        labels = []

        for imgs in blurred_images:
            img_results = []
            for img, _ in imgs:
                latent, _ = deblur(img)
                img_results.append(latent)
            results.append(img_results)

        for i, (imgs, truth) in enumerate(zip(results, truth_images)):
            print("Computing average psnr for image {}".format(i))
            values.append(get_average_psnr(imgs, truth))
            labels.append(str(i + 1))

        if not os.path.exists('results'):
            os.mkdir('results')
        with open('results/psnr_values.pkl', 'wb') as f:
            pickle.dump(values, f)
        with open('results/psnr_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    else:
        with open('results/psnr_values.pkl', 'rb') as f:
            values = pickle.load(f)
    
    x_axis = np.arange(4)
    plt.bar(x_axis - 0.2, values, 0.4, label='Ours')
    plt.bar(x_axis + 0.2, reported_psnr, 0.4, label='Paper\'s')
    plt.xticks(x_axis, x_axis + 1)
    plt.legend()
    plt.ylabel("Average PSNR")
    plt.xlabel("Image Number")
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    main()
