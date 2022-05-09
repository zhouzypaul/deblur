import os

import cv2
import numpy as np
from skimage.io import imread
from cv2 import filter2D


"""
this file handles parsing the datasets and getting the data from the online datasets
"""
def parse_ieee_dataset(image_path, kernel_path):
    """
    this function parses the dataset and returns a list of images
    args:
        image_path: the path to the images
        kernel_path: the path to the kernels
    return:
        a list of ground truth images, a list of blurred images
    """
    true = []
    blur = []
    for im in os.listdir(image_path):
        if im.endswith(".png"):
            im = imread(os.path.join(image_path, im))[:,:,:3]
            true.append(im)
        img_blurs = []
        for k_path in os.listdir(kernel_path):
            if k_path.endswith(".png"):
                kernel = imread(os.path.join(kernel_path, k_path), as_gray=True)
                blurred = filter2D(im, -1, kernel/np.sum(kernel))
                img_blurs.append((blurred, kernel))
        blur.append(img_blurs)
    return true, blur


def parse_kaggle_blur_data(image_path='data/kaggle_blur', num_images=10):
    """
    parse the kaggle blur data from:
    https://www.kaggle.com/datasets/kwentar/blur-dataset?resource=download

    only load `num_images` images
    """
    def _defocused_blur_img_path(sharp_image_file_name):
        stem = os.path.splitext(sharp_image_file_name)[0]
        extension = os.path.splitext(sharp_image_file_name)[1]
        new_stem = stem[:-1] + 'F'
        new_file = new_stem  + extension
        return os.path.join(defocused_blur_path, new_file)

    def _motion_blur_img_path(sharp_image_file_name):
        stem = os.path.splitext(sharp_image_file_name)[0]
        extension = os.path.splitext(sharp_image_file_name)[1]
        new_stem = stem[:-1] + 'M'
        new_file = new_stem + extension
        return os.path.join(motion_blur_path, new_file)
    
    def read_and_downsize_img(img_path, scale=0.1):
        img = imread(img_path)
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        return cv2.resize(img, (width, height))

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist. Please download the dataset from https://www.kaggle.com/datasets/kwentar/blur-dataset?resource=download. Put the dataset in data/kaggle_blur")

    defocused_blur_path = os.path.join(image_path, 'defocused_blurred')
    motion_blur_path = os.path.join(image_path, 'motion_blurred')
    sharp_path = os.path.join(image_path, 'sharp')

    all_images = os.listdir(sharp_path)
    chosen_images = [all_images[i] for i in range(num_images)]
    
    sharp_images = [read_and_downsize_img(os.path.join(sharp_path, im)) for im in chosen_images]
    defocused_blur_images = [read_and_downsize_img(_defocused_blur_img_path(im)) for im in chosen_images]
    motion_blur_images = [read_and_downsize_img(_motion_blur_img_path(im)) for im in chosen_images]
    return sharp_images, defocused_blur_images, motion_blur_images


if __name__ == "__main__":
    # for debugging purposes
    sharp, defocused, motion = parse_kaggle_blur_data()
    print(len(sharp))
    print(sharp)
