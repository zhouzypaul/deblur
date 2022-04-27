from skimage.io import imread
import glob, os
from cv2 import filter2D
import numpy as np
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
    ims = []
    for im in os.listdir(image_path):
        if im.endswith(".png"):
            ims.append(imread(os.path.join(image_path, im)))
    out = []
    for k_path in os.listdir(kernel_path):
        if k_path.endswith(".png"):
            kernel = imread(os.path.join(kernel_path, k_path))
            for im in ims:
                blurred = filter2D(im, -1, kernel/np.sum(kernel))
                out.append(blurred)
    return ims, out
