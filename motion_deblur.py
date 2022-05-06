import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d

from conjugate_gradient import conjugate_gradient
from get_data import parse_dataset
from pypher import otf2psf, psf2otf, zero_pad
import params as hp


kernel_size = (15, 15)


def init_kernel():
    '''
    initializes kernel
    '''
    return np.ones(kernel_size) / np.prod(kernel_size)


def generate_image_pyramid(y):
    '''
    Repeatedly downsamples blurred image with bilinear interpolation
    y: single image - numpy array
    '''
    img = y.copy()
    try:
        assert isinstance(img, np.ndarray)
    except AssertionError:
        img = np.array(img)
    image_pyramid = [img]

    # downsample for fixed number of layers
    for _ in range(5):
        layer = cv2.pyrDown(np.array(image_pyramid[-1]))
        layer = cv2.pyrUp(layer)
        image_pyramid.append(layer)

    return image_pyramid


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
    for y in reversed(image_pyramid):

        # TODO: potentially need to resize kernel, estimated latent, and blurred image?

        # perform Algorithm 2 to estimate intermediate estimated latent and blur kernel
        x = estimate_latent(y, k)
        k = estimate_kernel(x, y, hp.lmda, kernel_size)

        latent_imgs.append(x)

    # perform final deconvolve using final estimated kernel
    final_deblurred = estimate_latent(y, k)
    k = estimate_kernel(x, y, hp.lmda, kernel_size)
    latent_imgs.append(final_deblurred)

    return k, latent_imgs[-1]


def estimate_latent(blur_img, kernel):
    latent = blur_img
    out_shape = blur_img.shape

    # precompute
    fx = psf2otf(np.array([[1, -1]]), out_shape)
    fy = psf2otf(np.array([[1, -1]]).T, out_shape)
    ker = psf2otf(kernel, out_shape)
    ker_ker = np.abs(ker) ** 2
    nablas = np.abs(fx) ** 2 + np.abs(fy) ** 2
    ker_latent = np.conj(ker) * fft2(latent)

    beta = 2 * hp.lmda * hp.sigma
    while beta < hp.beta_max:
        u = solve_u(latent, beta)
        miu = 2 * hp.lmda

        while miu < hp.miu_max:
            fg = compute_fg(latent, miu)
            nom = ker_latent + beta * fft2(u) + miu * fft2(fg)
            denom = ker_ker + beta + miu * nablas
            latent = np.real(ifft2(nom / denom))
            miu *= 2

        beta *= 2

    return latent


def estimate_kernel(latent_img, blur_img, weight, psf_size):
    # derivative kernels
    dx = np.array([[-1, 1], [0, 0]])
    dy = np.array([[-1, 0], [1, 0]])

    lxf = fft2(convolve2d(latent_img, dx, mode='valid'))
    lyf = fft2(convolve2d(latent_img, dy, mode='valid'))

    bxf = fft2(convolve2d(blur_img, dx, mode='valid'))
    byf = fft2(convolve2d(blur_img, dy, mode='valid'))

    b_f = np.conj(lxf) * bxf + np.conj(lyf) * byf
    b = np.real(otf2psf(b_f, psf_size))

    p = {}
    p['m'] = np.conj(lxf) * lxf + np.conj(lyf) * lyf
    p['img_size'] = np.shape(bxf)
    p['psf_size'] = psf_size
    p['lmda'] = weight

    psf = np.ones(psf_size) / np.prod(psf_size)
    psf = conjugate_gradient(psf, b, 20, 1e-5, compute_ax, p)
    
    psf = np.where(psf >= 0.05 * np.max(psf), psf, 0)
    psf /= np.sum(psf)
    return psf


def solve_u(latent, beta):
    threshold = hp.lmda * hp.sigma / beta
    return np.where(latent ** 2 >= threshold, latent, 0)

def solve_g(h, v, miu):
    condition = h ** 2 + v ** 2 >= hp.lmda / miu
    return np.where(condition, h, 0), np.where(condition, v, 0)

def compute_fg(latent, miu):
    # compute horizontal and vertical gradients to solve for g
    h_diff = latent[:, 0] - latent[:, -1]
    h = np.hstack((np.diff(latent, axis=1), h_diff[:, None]))
    v_diff = latent[0, :] - latent[-1, :]
    v = np.vstack((np.diff(latent, axis=0), v_diff))
    gh, gv = solve_g(h, v, miu)

    # compute horizontal and vertical components in FG
    gh_diff = gh[:, -1] - gh[:, 0]
    h_component = np.hstack((gh_diff[:, None], -np.diff(gh, axis=1)))
    gv_diff = gv[-1, :] - gv[0, :]
    v_component = np.vstack((gv_diff, -np.diff(gv, axis=0)))

    return h_component + v_component


def compute_ax(x, p):
    xf = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * xf, p['psf_size'])
    y = y + p['lmda'] * x
    return y


def show_latent_image(data_path='data/ieee2016/text-images/'):
    images_path = data_path + 'gt_images'
    kernels_path = data_path + 'kernels'
    clear, blur, kernels = parse_dataset(images_path, kernels_path)

    fig = plt.figure()

    ker_ind = 3
    img = 13
    img_ind = 8 * img + ker_ind
    res = estimate_latent(blur[img_ind], kernels[ker_ind])

    fig.add_subplot(1, 3, 1)
    plt.imshow(blur[img_ind], cmap='gray')
    fig.add_subplot(1, 3, 2)
    plt.imshow(kernels[ker_ind], cmap='gray')
    fig.add_subplot(1, 3, 3)
    plt.imshow(res, cmap='gray')
    plt.show()


def visualize_original_blurred_deblurred(original, blurred, deblurred):
    fig = plt.figure()
    fig.add_subplot(3, 1, 1)
    plt.title('Original')
    plt.imshow(original, cmap='gray')
    fig.add_subplot(3, 1, 2)
    plt.title('Blurred')
    plt.imshow(blurred, cmap='gray')
    fig.add_subplot(3, 1, 3)
    plt.title('Deblurred')
    plt.imshow(deblurred, cmap='gray')
    plt.show()
    plt.close()


def main():
    """
    interleave kernel and latent image estimation
    load the original images and deblur them, and plot them together
    """
    ground_truth_images, blurred_images, kernels = parse_dataset(image_path='data/ieee2016/text-images/gt_images', kernel_path='data/ieee2016/text-images/kernels')  # there should be 15 of them
    for ground_truth, blurs in zip(ground_truth_images, blurred_images):
        for blur in blurs:
            kernel, res = deblur(blur)
            visualize_original_blurred_deblurred(ground_truth, blur, res)

if __name__ == '__main__':
    main()
