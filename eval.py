import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib import pyplot as plt

from motion_deblur import deblur
from get_data import parse_ieee_dataset

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
    image_path = 'data/ieee2016/text-images/gt_images'
    kernel_path = 'data/ieee2016/text-images/kernels'
    truth_images, blurred_images = parse_ieee_dataset(image_path, kernel_path, as_gray=True)
    results = []
    values = []
    labels = []
    reported_psnr = [27.5, 29, 23, 24, 27, 28.5, 30.5, 28, 26.5, 29.5, 26, 38, 31, 31, 30]

    for imgs in blurred_images:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(deblur, img) for img, _ in imgs]
            img_results = [future.result()[0] for future in futures]
        results.append(img_results)

    # save the result to be safe
    if not os.path.exists('results'):
        os.mkdir('results')
    with open('results/psnr_results', 'wb') as f:
        pickle.dump(results, f)

    for i, (imgs, truth) in enumerate(zip(results, truth_images)):
        print("Computing average psnr for image {}".format(i))
        values.append(get_average_psnr(imgs, truth))
        labels.append(str(i + 1))

    # save the results
    with open('results/psnr_values.pkl', 'wb') as f:
        pickle.dump(values, f)

    plt.bar(labels - 0.2, values, 0.4, label='Ours')
    plt.bar(labels + 0.2, reported_psnr, 0.4, label='Pan et al.')
    plt.legend()
    plt.ylabel("Average PSNR")
    plt.xlabel("Image Number")
    plt.savefig("results/psnr.png")


if __name__ == "__main__":
    np.random.seed(0)
    main()
