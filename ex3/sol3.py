__author__ = 'Avichai'

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import signal as sig
from scipy.ndimage.filters import convolve
import os



IDENTITY_KERNEL_SIZE = 1
BINOMIAL_MAT = [0.5, 0.5]

GRAY = 1
RGB = 2
NORM_PIX_FACTOR = 255

MIN_DIM_PYR = 16




#todo not in use for noe

IMAGE_DIM = 2
DER_VEC = [1, 0, -1]
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

def getGaussVec(kernel_size):
    '''
    gets the gaussian vector in the length of the kernel size
    :param kernel_size: the length of the wished kernel
    :return: the 1d vector we want
    '''
    if kernel_size == IDENTITY_KERNEL_SIZE:
        return [1]
    return sig.convolve(BINOMIAL_MAT, getGaussVec(kernel_size-1))


def getGaussMat(kernel_size):
    '''
    getting a gaussian kernel of size kernel_sise^2
    :param kernel_size: the size of the wished kernel in
    each dimension (an odd integer)
    :return: gaussian kernel of size kernel_sise^2
    '''
    '''geeting the blure vec in 1d'''
    gaussVec = getGaussVec(kernel_size)
    '''creating the 2d kernel'''
    gaussAsMat = np.array(gaussVec)
    gaussMat = sig.convolve2d(gaussAsMat.reshape(kernel_size, 1),
                             gaussAsMat.reshape(1, kernel_size))
    return gaussMat

def reduceIm(currIm, gaussFilter):
    blurIm = convolv(currIm, gaussFilter)
    pass

def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: Gaussian pyramid as standard python array
    '''
    gaussFilter = getGaussMat(filter_size)
    gaussPyr = [im]
    currIm = im

    # todo add braking when size of im is less than 16 pix
    for i in range (max_levels):
        if currIm.shape[0]//2 < MIN_DIM_PYR or currIm.shape[1]//2 < MIN_DIM_PYR:
            break
        currIm = reduceIm(currIm, gaussFilter)
        gaussPyr = gaussPyr + currIm
    pass


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: Laplacian pyramid as standard python array
    '''
    pass