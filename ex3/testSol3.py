from ex3 import sol3
import matplotlib.pyplot as plt

def testGaussPyr():
    im = sol3.read_image('external/jerusalem.jpg', 1)
    sol3.imdisplay('external/jerusalem.jpg', 1)
    # pyr, filter_vec = sol3.build_gaussian_pyramid(im, max_levels, filter_size)









def main():
    try:
        for test in [testGaussPyr]:
            test()
    except Exception as e:
        print("Failed test due to: {0}".format(e))
        exit(-1)

if __name__ == '__main__':
    main()

