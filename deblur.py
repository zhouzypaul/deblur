"""
This file handles the debluring of images
this is where the magic comes together
"""
import numpy as np
from numpy.fft import fft2, ifft2
import sklearn
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve
from scipy.fftpack import fftn, ifftn

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


# ==============================================================================
num_iters = 5
gamma = 1.0
kernel_size = (15, 15)

# TODO: decide on variable names, for now use paper conventions:
# x -> latent image
# y -> blur image
# k -> blur kernel


def deblur(y):
    """
    this function deblurs an image by eestimating the blur kernel using 
    a coarse-to-fine image pyramid approach

    this is section 3.2 in the paper
    args:
        y: the blur image
    return:
        final blur kernel and list of estimated latent images
    """

    # image pyramid generated by downsampling blur_img
    image_pyramid = generate_image_pyramid(y)

    # previous blur_kernel
    k = init_kernel()

    # intermediate latent images
    latent_imgs = []

    # coarse to fine
    for y in image_pyramid.reverse():

        # TODO: potentially need to resize kernel, estimated latent, and blurred image?

        # perform Algorithm 2 to estimate intermediate estimated latent and blur kernel
        k, x = estimate_blur_kernel(y, k)

        latent_imgs.append(x)

    # perform final deconvolve using final estimated kernel
    k, final_deblurred = estimate_blur_kernel(y, k)
    latent_imgs.append(final_deblurred)

    return k, latent_imgs


def estimate_blur_kernel(y, k):
    """
    this function estimates the blur kernel using an interative process as
    outlined in Algorithm 2 of the paper
    args:
        y: the blur image
        k: the blur kernel estimate
    return:
        blur kernel and intermediate latent image
    """
    # intermediate latent image
    x = None

    # preset number of iterations, paper uses 5
    for _ in range(5):

        # solve for x using Algorithm 1
        x = estimating_latent_image_with_blur_kernel(y, k)

        # solve for k, update kernel estimate eq. (12)
        # use FFT to estimate blur kernel, look at 'estimate_psf'
        k = solve_kernel(y, x)

        # update lambda hyperparameter
        lmda = max(lmda / 1.1, 1e-4)

    return k, x


def generate_image_pyramid(y):
    '''
    Repeatedly downsamples blurred image with bilinear interpolation
    '''
    img = y.copy()
    image_pyramid = [img]

    # downsample for fixed number of layers
    for _ in range(5):
        layer = cv2.pyrDown(image_pyramid[-1])
        layer = cv2.pyrUp(layer)
        image_pyramid.append(layer)

    return image_pyramid


def init_kernel():
    '''
    TODO: initializes kernel
    '''
    return np.ones(kernel_size)


def solve_kernel(y, x):
    '''
    this function estimates the blur kernel efficiently in gradient space 
    using FFT as described in https://dl.acm.org/doi/pdf/10.1145/1618452.1618491 
    TODO: not done, paul pls help
    '''
    dx = [[-1, 1], [0, 0]]
    dy = [[-1, 0], [1, 0]]

    latent_x = convolve(y, dx)
    latent_y = convolve(y, dy)

    blurred_x = convolve(x, dx)
    blurred_y = convolve(x, dy)

    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)

    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    b_f = (np.conj(latent_xf) * blurred_xf) + (np.conj(latent_yf) * blurred_yf)
    b = np.real(otf2psf(b_f, kernel_size))

    p.m = (np.conj(latent_xf) * latent_xf) + (np.conj(latent_yf) * latent_yf)
    %p.img_size = size(blurred)
    p.img_size = size(blurred_xf)
    p.psf_size = psf_size
    p.lambda = weight

    psf = ones(psf_size) / prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, @ compute_Ax, p)

    psf(psf < max(psf(:))*0.05) = 0
    psf = psf / sum(psf(: ))


# ===== TAKEN FROM pypher ====================================


def otf2psf(otf, psf_size):
    # calculate psf from otf with size <= otf size

    if otf.any():  # if any otf element is non-zero
        # calculate psf
        psf = ifftn(otf)
        # this condition depends on psf size
        num_small = np.log2(otf.shape[0])*4*np.spacing(1)
        if np.max(abs(psf.imag))/np.max(abs(psf)) <= num_small:
            psf = psf.real

        # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0]/2)), axis=0)
        psf = np.roll(psf, int(np.floor(psf_size[1]/2)), axis=1)

        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else:  # if all otf elements are zero
        psf = np.zeros(psf_size)
    return psf

# ==============================================================================


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
        blur_kernel, restored_img = deblur(img)
        for latent_img in restored_img:
            plt.imshow(latent_img)
        plt.show()
        plt.close()

    # test for image pyramid
    # img = cv2.imread('Alan_Kumon_W2_2019.png')
    # cv2.imshow("original image", img)

    # pyramid = generate_image_pyramid(img)
    # for i, img in enumerate(pyramid):
    #     cv2.imshow(str(i) + "image", img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
