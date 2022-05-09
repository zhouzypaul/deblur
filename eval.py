from matplotlib import pyplot as plt
from get_data import parse_ieee_dataset
import numpy as np

from motion_deblur import deblur

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
    def psnr(result, truth):
        mse = np.mean((truth.astype(np.uint8) - result.astype(np.uint8)) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(255 / np.sqrt(mse))

    values = []
    for img in results:
        values.append(psnr(img, truth))
    return np.mean(np.asarray(values))


def main():
    """
    plot the results of average psnr
    """
    image_path = 'data/ieee2016/text-images/gt_images'
    kernel_path = 'data/ieee2016/text-images/kernels'
    truth_images, blurred_images = parse_ieee_dataset(image_path, kernel_path)
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

    plt.bar(labels, values)
    plt.ylabel("Average PSNR")
    plt.xlabel("Image Number")
    plt.show()


if __name__ == "__main__":
    main()
