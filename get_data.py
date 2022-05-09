import os

import numpy as np
from skimage.io import imread
from cv2 import filter2D


"""
this file handles parsing the datasets and getting the data from the online datasets
"""
def parse_dataset(image_path, kernel_path):
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
