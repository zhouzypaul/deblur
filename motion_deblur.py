import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d

from conjugate_gradient import conjugate_gradient
from get_data import parse_dataset
from pypher import otf2psf, psf2otf
import params as hp

"""
Deblurs images based on iterative latent image and blur kernel estimation
"""


def deblur(blur_img):
    """
    this function deblurs an image by eestimating the blur kernel using 
    a coarse-to-fine image pyramid approach
    this is section 3.2 in the paper
    args:
        y: the blur image
    return:
        final blur kernel and list of estimated latent images
    """
    hp.lmda = 4e-3

    # image pyramid generated by downsampling blur_img
    image_pyramid = generate_image_pyramid(blur_img)

    # previous blur_kernel
    k = init_kernel()

    # intermediate latent images
    latent_imgs = []

    # coarse to fine
    for y in reversed(image_pyramid):

        # perform Algorithm 2 to estimate intermediate estimated latent and blur kernel
        k, x = estimate_blur_kernel(y, k)

        latent_imgs.append(x)

    # perform final deconvolve using final estimated kernel
    k, final_deblurred = estimate_blur_kernel(y, k)
    latent_imgs.append(final_deblurred)

    k = cv2.rotate(k, cv2.ROTATE_180)

    return final_deblurred, k


def estimate_blur_kernel(blur_img, kernel):
    """
    this function estimates the blur kernel using an interative process as
    outlined in Algorithm 2 of the paper
    args:
        blur_img: the blur image
        kernel: the blur kernel estimate
    return:
        blur kernel and intermediate latent image
    """
    # intermediate latent image
    x = None
    k = kernel.copy()

    # preset number of iterations, paper uses 5
    for _ in range(3):

        # solve for x using Algorithm 1
        x = solve_latent(blur_img, k)

        # solve for k, update kernel estimate eq. (12)
        k = solve_kernel(x, blur_img, hp.lmda)

        # update lambda hyperparameter
        hp.lmda = max(hp.lmda / 1.1, 1e-4)

    return k, x


def solve_latent(blur_img, kernel):
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


def solve_kernel(latent_img, blur_img, weight):
    """
    this function estimates the blur kernel using an interative process as
    outlined in Algorithm 2 of the paper
    args:
        y: the blur image
        k: the blur kernel estimate
    return:
        blur kernel and intermediate latent image
    """

    # derivative kernels
    dx = np.array([[-1, 1], [0, 0]])
    dy = np.array([[-1, 0], [1, 0]])

    lxf = fft2(convolve2d(latent_img, dx, mode='valid'))
    lyf = fft2(convolve2d(latent_img, dy, mode='valid'))

    bxf = fft2(convolve2d(blur_img, dx, mode='valid'))
    byf = fft2(convolve2d(blur_img, dy, mode='valid'))

    b_f = np.conj(lxf) * bxf + np.conj(lyf) * byf
    b = np.real(otf2psf(b_f, hp.kernel_size))

    p = {}
    p['m'] = np.conj(lxf) * lxf + np.conj(lyf) * lyf
    p['img_size'] = np.shape(bxf)
    p['kernel_size'] = hp.kernel_size
    p['lmda'] = weight

    psf = np.ones(hp.kernel_size) / np.prod(hp.kernel_size)
    psf = conjugate_gradient(psf, b, 20, 1e-5, compute_ax, p)

    psf = np.where(psf >= 0.05 * np.max(psf), psf, 0)
    psf /= np.sum(psf)
    return psf


# ==============================================================================


# HELPERS ======================================================================


def init_kernel():
    '''
    Initializes kernel estimate

    args:
        kernel_size: size of kernel to estimate
    returns:
        kernel: 2d np array
    '''
    n, _ = hp.kernel_size
    kernel = np.zeros(hp.kernel_size)
    mid = (n - 1) // 2

    # set middle two pixels to 1/2
    kernel[mid, mid:mid + 2] = 1 / 2
    return kernel


def generate_image_pyramid(y):
    '''
    Repeatedly downsamples blurred image with bilinear interpolation

    args:
        y: single image - numpy array
    returns:
        list of images - list of numpy arrays
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


def solve_u(latent, beta):
    '''
    Solve intermediate step in Algorithm 1
    '''
    threshold = hp.lmda * hp.sigma / beta
    return np.where(latent ** 2 >= threshold, latent, 0)


def solve_g(h, v, miu):
    '''
    Solve intermediate step in Algorithm 1
    '''
    condition = h ** 2 + v ** 2 >= hp.lmda / miu
    return np.where(condition, h, 0), np.where(condition, v, 0)


def compute_fg(latent, miu):
    '''
    Compute intermediate step in Algorithm 1
    '''
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
    '''
    Computes Ap term for system of linear equations

    args:
        x: x from Ax=b
        p: dict of function parameterss
    returns:
        Axpterm
    '''
    xf = psf2otf(x, p['img_size'])
    ap = otf2psf(p['m'] * xf, p['kernel_size'])
    ap += p['lmda'] * x
    return ap


def visualize_results(original, blurred, deblurred, kernel, est_kernel):
    fig = plt.figure()
    fig.add_subplot(2, 3, 1)
    plt.title('Original')
    plt.imshow(original, cmap='gray')
    fig.add_subplot(2, 3, 2)
    plt.title('Blurred')
    plt.imshow(blurred, cmap='gray')
    fig.add_subplot(2, 3, 3)
    plt.title('Deblurred')
    plt.imshow(deblurred, cmap='gray')
    fig.add_subplot(2, 3, 4)
    plt.title('Kernel')
    plt.imshow(kernel, cmap='gray')
    fig.add_subplot(2, 3, 5)
    plt.title('Estimated Kernel')
    plt.imshow(est_kernel, cmap='gray')
    plt.tight_layout()
    plt.show()
    plt.close()


# ==============================================================================


def main():
    """
    interleave kernel and latent image estimation
    load the original images and deblur them, and plot them together
    """
    image_path = 'data/ieee2016/text-images/gt_images'
    kernel_path = 'data/ieee2016/text-images/kernels'
    ground_truth_images, blurred_images = parse_dataset(
        image_path, kernel_path)  # there should be 15 of them

    for ground_truth, blurs in zip(ground_truth_images, blurred_images):
        rind = np.random.randint(len(blurs))
        blur_img, kernel = blurs[rind]
        latent, est_kernel = deblur(blur_img)
        visualize_results(ground_truth, blur_img, latent, kernel, est_kernel)


if __name__ == '__main__':
    main()
