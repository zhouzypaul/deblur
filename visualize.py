from matplotlib import pyplot as plt


"""
visualization methods
"""
def visualize_text_deblurs(original, blurred, deblurred, kernel, est_kernel, artifact_removed=None, save_path=None):
    """
    visualize the debluring results for the text dataset and the estimated kernels
    """
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
    plt.title('Groud Truth Blur Kernel')
    plt.imshow(kernel, cmap='gray')

    fig.add_subplot(2, 3, 5)
    plt.title('Estimated Kernel')
    plt.imshow(est_kernel, cmap='gray')

    if artifact_removed is not None:
        fig.add_subplot(2, 3, 6)
        plt.title('Artifact Removed')
        plt.imshow(artifact_removed)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def visualize_kaggle_deblurs(sharp, defocused, motioned, defocused_latent, motioned_latent, save_path=None):
    """
    visualize the debluring results for the kaggle dataset
    """
    fig = plt.figure()
    fig.add_subplot(2, 3, 1)
    plt.title('Original')
    plt.imshow(sharp)

    fig.add_subplot(2, 3, 2)
    plt.title('Defocus Blur')
    plt.imshow(defocused)

    fig.add_subplot(2, 3, 3)
    plt.title('Motion Blur')
    plt.imshow(motioned)

    fig.add_subplot(2, 3, 4)
    plt.title('Defocus Blur Deblurred')
    plt.imshow(defocused_latent)

    fig.add_subplot(2, 3, 5)
    plt.title('Motion Blur Deblurred')
    plt.imshow(motioned_latent)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()