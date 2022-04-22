"""
this file handles the evaluation of the debluring methods
"""
from matplotlib import pyplot as plt
from get_data import parse_dataset


def get_average_psnr(img_list):
    """
    this function calculates the average psnr of a list of images
    args:
        img_list: a list of images (should be 15 of them from the paper's dataset)
    return:
        the average psnr of the list of images
    """
    # TODO: Alan
    pass


def main(data_path='data/original_images/'):
    """
    plot the results of average psnr
    """
    blurred_images = parse_dataset(data_path)  # there should be 15 of them
    res = get_average_psnr(blurred_images)
    


if __name__ == "__main__":
    main()
