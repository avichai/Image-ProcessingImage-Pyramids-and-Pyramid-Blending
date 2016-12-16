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
ROWS = 0
COLS = 1
LARGEST_IM_INDEX = 0



# todo not in use for noe

IMAGE_DIM = 2
DER_VEC = [1, 0, -1]
DIM_RGB = 3
MAX_PIX_VALUE = 256
MIN_PIX_VALUE = 0


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
    if (representation == GRAY):
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
    return sig.convolve(BINOMIAL_MAT, getGaussVec(kernel_size - 1))


def getImAfterBlur(im, Filter, filter_size):
    '''
    return the image after row and col blur
    :param im: the image to blur
    :param Filter: the filter to blur with
    :return: blurred image
    '''
    filterAsMat = np.array(Filter).reshape(1, filter_size)
    # todo choose which convolve do I prefer
    # blurXIm = convolve(im, filterAsMat, mode='reflect')
    blurXIm = convolve(im, filterAsMat, mode='constant', cval=0.0)
    # blurIm = convolve(blurXIm, filterAsMat.transpose(), mode='reflect')
    blurIm = convolve(blurXIm, filterAsMat.transpose(),
                      mode='constant', cval=0.0).astype(np.float32)
    return blurIm


def reduceIm(currIm, gaussFilter, filter_size):
    '''
    reduce an image
    :param currIm: the image to reduce by 4
    :param gaussFilter: the filter to blur with the image before reduce
    :param filter_size: the size of the filter
    :return: the reduced image
    '''
    blurIm = getImAfterBlur(currIm, gaussFilter, filter_size)
    reducedImage = blurIm[::2, ::2]
    return reducedImage.astype(np.float32)


def expandIm(currIm, gaussFilterForExpand, filter_size):
    '''
    expand an image
    :rtype : np.float32
    :param currIm: the image to expand by 4
    :param gaussFilterForExpand: the filter to blur with the expand image
    :param filter_size: the size of the filter
    :return: an expand image
    '''
    expandImage = np.zeros((2 * currIm.shape[0], 2 * currIm.shape[1]))
    expandImage[::2, ::2] = currIm
    expandRes = getImAfterBlur(expandImage, gaussFilterForExpand, filter_size)
    return expandRes.astype(np.float32)


def getNumInInPyr(im, max_levels):
    '''
    return maximum number of images in pyramid
    :param im: tne original image
    :param max_levels: an initial limitation
    :return: the real limitation
    '''
    numRows = im.shape[ROWS]
    numCols = im.shape[COLS]

    limRows = np.floor(np.log2(numRows)) - 3
    limCols = np.floor(np.log2(numCols)) - 3
    numImInPyr = np.uint8(np.min([max_levels, limCols, limRows]))
    return numImInPyr


def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: Gaussian pyramid as standard python array
    '''
    numImInPyr = getNumInInPyr(im, max_levels)
    gaussFilter = getGaussVec(filter_size)

    gaussPyr = [im]
    currIm = im

    # todo add braking when size of im is less than 16 pix
    for i in range(1, numImInPyr):
        currIm = reduceIm(currIm, gaussFilter, filter_size)
        gaussPyr.append(currIm)
    return gaussPyr, gaussFilter


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return: Laplacian pyramid as standard python array and the filter vec
    '''
    gaussFilter = getGaussVec(filter_size)
    laplacianPyr = []

    gaussPyr = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    numImInPyr = len(gaussPyr)

    for i in range(numImInPyr - 1):
        laplacianPyr.append(gaussPyr[i] - expandIm(
            gaussPyr[i + 1], 2 * gaussFilter, filter_size))
    laplacianPyr.append(gaussPyr[numImInPyr - 1])
    return laplacianPyr, gaussFilter


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    reconstruction of an image from its Laplacian Pyramid
    :param lpyr: Laplacian pyramid
    :param filter_vec: the filter that was used in order to
            construct the pyramid
    :param coeff: the coefficient of each image in the pyramid
    :return: reconstruction of an image from its Laplacian Pyramid
    '''
    numIm = len(lpyr)
    numCoe = len(coeff)
    if numIm != numCoe:
        '''invalid input'''
        return
    gni = lpyr[numIm - 1]
    for i in range(numIm - 1):
        gni = expandIm(gni, 2 * filter_vec, len(filter_vec)) + (
            lpyr[numIm - 1 - i - 1] * coeff[len(coeff) - 1 - i])
    return gni.astype(np.float32)


def strechIm(im, newMin, newMax):
    """
    strech the image to [newMin, newMax]
    :param newMax: max vlue to stretch to
    :param newMin: min value to stretch to
    :param im: float 32 image
    :return: stretched im
    """
    inMin = np.min(im)
    inMax = np.max(im)
    stretchedIm = (im - inMin) * ((newMax - newMin) / (inMax - inMin)) + newMin
    return stretchedIm


def render_pyramid(pyr, levels):
    '''
    creates a single black image in which the pyramid levels of the
        given pyramid pyr are stacked horizontally
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: number of levels to present in the result
    :return: single black image in which the pyramid levels of the
        given pyramid pyr are stacked horizontally
    '''
    levels = min(levels, len(pyr))
    numRows = pyr[LARGEST_IM_INDEX].shape[ROWS]
    numCols = pyr[LARGEST_IM_INDEX].shape[COLS]
    for i in range(1, levels):
        numCols += pyr[i].shape[COLS]
    pyrIm = np.zeros([numRows, numCols])
    curPlace = 0
    for i in range(levels):
        stretchedIm = strechIm(pyr[i], 0, 1)
        rows = stretchedIm.shape[ROWS]
        cols = stretchedIm.shape[COLS]
        pyrIm[:rows, curPlace:curPlace + cols] = stretchedIm
        curPlace = curPlace + cols
    return pyrIm.astype(np.float32)


def display_pyramid(pyr, levels):
    '''
    display at max the amount of levels from the pyr
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: number of levels to present in the result
    '''
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show(block=True)