__author__ = 'Avichai'

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import signal as sig
import os


IMAGE_DIM = 2
IDENTITY_KERNEL_SIZE = 1
BINOMIAL_MAT = [0.5, 0.5]
DER_VEC = [1, 0, -1]

GRAY = 1
RGB = 2
NORM_PIX_FACTOR = 255
DIM_RGB = 3
MAX_PIX_VALUE = 256
MIN_PIX_VALUE = 0
Y = 0
ROWS = 0
COLS = 1



def read_image(filename, representation):
    """this function reads a given image file and converts it into a given
    representation:
    filename - string containing the image filename to read.
    representation - representation code, either 1 or 2 defining if the
                     output should be either a grayscale image (1) or an
                     RGB image (2).
    output - the image in the given representation when the pixels are
             of type np.float32 and normalized"""
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        return
    im = imread(filename)
    if im.dtype == np.float32:
        '''I don't handle this case, we asume imput in uint8 format'''
        return
    if representation == GRAY:
        im = rgb2gray(im).astype(np.float32)
        return im
    im = im.astype(np.float32)
    im /= NORM_PIX_FACTOR
    return im


def imdisplay(filename, representation):
    """this function display a given image file and in the given
    representation:
    filename - string containing the image filename to read.
    representation - representation code, either 1 or 2 defining if the
                     output should be either a grayscale image (1) or an
                     RGB image (2)."""
    im = read_image(filename, representation)
    plt.figure()
    if(representation == GRAY):
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.show(block=True)


def build_gaussian_pyramid(im, max_levels, filter_size):
    pass