from ex3 import sol3
import matplotlib.pyplot as plt
import numpy as np


def testGaussPyr():
    NUM_PIC = 4
    im = sol3.read_image('external/jerusalem.jpg', 1)
    # sol3.imdisplay('external/jerusalem.jpg', 1)
    im = im[:2 ** (np.uint(np.floor(np.log2(im.shape[0])))),
         :2 ** (np.uint(np.floor(np.log2(im.shape[1]))))]
    pyr, filter_vec = sol3.build_gaussian_pyramid(im, NUM_PIC, 11)
    print(filter_vec)

    plt.figure()
    for i in range(len(pyr)):
        plt.subplot(1, NUM_PIC, 1 + i)
        plt.imshow(pyr[i], cmap=plt.cm.gray)
    plt.show(block=True)


def testLapPyr():
    NUM_PIC = 4
    im = sol3.read_image('external/jerusalem.jpg', 1)
    im = im[:2 ** (np.uint(np.floor(np.log2(im.shape[0])))),
         :2 ** (np.uint(np.floor(np.log2(im.shape[1]))))]
    pyr, filter_vec = sol3.build_laplacian_pyramid(im, NUM_PIC, 3)
    print(filter_vec)

    plt.figure()
    for i in range(len(pyr)):
        plt.subplot(1, NUM_PIC, 1 + i)
        plt.imshow(pyr[i], cmap=plt.cm.gray)
    plt.show(block=True)


def testReconstract():
    NUM_PIC = 7
    im = sol3.read_image('external/jerusalem.jpg', 1)
    # im = sol3.read_image('external/monkey.jpg', 1)
    # im = sol3.read_image('external/LowContrast.jpg', 1)
    im = im[:2 ** (np.uint(np.floor(np.log2(im.shape[0])))),
         :2 ** (np.uint(np.floor(np.log2(im.shape[1]))))]
    lpyr, filter_vec = sol3.build_laplacian_pyramid(im, NUM_PIC, 7)
    img = sol3.laplacian_to_image(lpyr, filter_vec, [1, 1, 1, 1, 1])

    print(filter_vec.shape)
    print(lpyr[4].shape)
    print(len(lpyr))

    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(img, cmap=plt.cm.gray)

    print(np.mean(np.abs(img - im)))
    print(np.max(np.abs(img - im)))
    plt.show(block=True)


def testRenderPyr():
    NUM_PIC = 5
    im = sol3.read_image('external/jerusalem.jpg', 1)
    # sol3.imdisplay('external/jerusalem.jpg', 1)
    im = im[:2 ** (np.uint(np.floor(np.log2(im.shape[0])))),
         :2 ** (np.uint(np.floor(np.log2(im.shape[1]))))]
    pyrg, filter_vec1 = sol3.build_gaussian_pyramid(im, NUM_PIC, 11)
    pyrl, filter_vec2 = sol3.build_laplacian_pyramid(im, NUM_PIC, 11)

    resg = sol3.render_pyramid(pyrg, len(pyrg))


    plt.figure()
    plt.imshow(resg, cmap=plt.cm.gray)

    resl = sol3.render_pyramid(pyrl, 10)
    plt.figure()
    plt.imshow(resl, cmap=plt.cm.gray)
    plt.show(block=True)
    # print(resl)


def testDispPyr():
    NUM_PIC = 5
    im = sol3.read_image('external/jerusalem.jpg', 1)
    # sol3.imdisplay('external/jerusalem.jpg', 1)
    im = im[:2 ** (np.uint(np.floor(np.log2(im.shape[0])))),
         :2 ** (np.uint(np.floor(np.log2(im.shape[1]))))]
    pyrg, filter_vec1 = sol3.build_gaussian_pyramid(im, NUM_PIC, 11)
    pyrl, filter_vec2 = sol3.build_laplacian_pyramid(im, NUM_PIC, 11)
    sol3.display_pyramid(pyrg, 10)
    sol3.display_pyramid(pyrl, 10)


def testBlend():
    max_levels = 5
    filter_size_im = 5
    filter_size_mask = 5
    mask32 = sol3.read_image('external/bonus/maskCamelBool.jpg', 1)
    mask = mask32.astype(np.bool)
    im1 = sol3.read_image('external/bonus/camel.jpg', 1)
    im2 = sol3.read_image('external/bonus/view.jpg', 1)
    im_blend = sol3.pyramid_blending(im1, im2, mask, max_levels,
                                      filter_size_im, filter_size_mask)

    plt.figure()
    plt.imshow(im_blend, cmap=plt.cm.gray)
    plt.show(block=True)


def testBlendExample():
    import time as time
    t1 = time.time()
    im1, im2, mask, im_blend = sol3.blending_example1()
    print (time.time()-t1)
    t2 = time.time()
    im3, im4, mask1, im_blend1 = sol3.blending_example2()
    print(time.time() - t2)
    plt.figure()
    plt.imshow(im_blend)
    plt.figure()
    plt.imshow(im_blend1)

    plt.show()


# tests:

# testGaussPyr
# testLapPyr
# testReconstract
# testRenderPyr
# testDispPyr
# testBlend
# testBlendExample


def main():
    try:
        for test in [testBlendExample]:
            test()
    except Exception as e:
        print("Failed test due to: {0}".format(e))
        exit(-1)


if __name__ == '__main__':
    main()

