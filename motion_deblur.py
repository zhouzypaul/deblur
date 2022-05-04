from ast import parse
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from get_data import parse_dataset
from pypher import psf2otf, zero_pad
import params as hp

def deblur(img):
    return img


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
            denom = ker_ker + miu * nablas + beta
            latent = np.real(ifft2(nom / denom))
            miu *= 2

        beta *= 2

    return latent


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

def main():
    clear, blur = parse_dataset(image_path='data/ieee2016/text-images/gt_images', kernel_path='data/ieee2016/text-images/kernels')
    # for img in blur:
    #     kernel, res = deblur(img)
    #     plt.imshow(res, cmap='gray')
    #     plt.show()

    res = estimate_latent(blur[0], np.ones((15, 15)))
    plt.imshow(res, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
