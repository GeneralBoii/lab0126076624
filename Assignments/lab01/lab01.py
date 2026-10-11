# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys

# Allow helpers package to be found when this module is imported standalone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import load_image, get_data_path

SPHINX_IMAGE = "2560px-Great_Sphinx_of_Giza_-_20080716a.jpg"

def read_image():
    """Load and preprocess the Sphinx image: scale by 0.5 and convert to greyscale."""
    image_path = get_data_path(SPHINX_IMAGE)
    return load_image(image_path, scale_factor=2, as_gray=True)

# Skeleton Code will generate a 1D Gaussian Kernel for given sigma.
# You need to implement 2D convolution as efficiently as possible.
class GaussianFilt:
    def __init__(self, sigma):
        self.sigma = sigma

    def gauss_kernel(self):
        """Generate a normalised 1D Gaussian kernel of appropriate size for self.sigma.

        Returns:
            gauss_1d (ndarray): Shape (k_size, 1) — column vector kernel.
        """
        if self.sigma <= 0:
            raise ValueError("Sigma must be positive")

        radius = int(np.ceil(3 * self.sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(x ** 2) / (2 * self.sigma ** 2))

        # Normalise so the kernel sums to 1
        kernel /= kernel.sum()

        return kernel.reshape(-1, 1)

    def my_conv_method(self, image):
        """Perform separable 2D Gaussian convolution via two 1D matrix-multiply passes.

        Args:
            image (ndarray): 2D greyscale image (H, W), any numeric dtype.

        Returns:
            conv_img (ndarray): Blurred image of same shape, dtype uint8.
        """
        """Perform separable 2D Gaussian convolution via two 1D matrix-multiply passes."""
        
        kernel = self.gauss_kernel().ravel()

        # Convolve each row using the 1D kernel
        horizontal = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"),
            axis=1,
            arr=image.astype(np.float64),
        )

        # Convolve each column of the result
        vertical = np.apply_along_axis(
            lambda column: np.convolve(column, kernel, mode="same"),
            axis=0,
            arr=horizontal,
        )

        return vertical.astype(np.uint8)
