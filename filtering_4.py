import numpy as np


# ----------------------------------------------------------------
def norm_1(img: np.array):
    res = 0.0
    x_range = img.shape[1]
    y_range = img.shape[0]
    for x in range(0, x_range):
        for y in range(0, y_range):
            res += (float(img[y][x])) ** 2
    return res


def norm_2(img: np.array, img_old: np.array):
    res = 0.0
    x_range = img.shape[1]
    y_range = img.shape[0]
    for x in range(0, x_range):
        for y in range(0, y_range):
            res += (float(img[y][x]) - float(img_old[y][x])) ** 2
    return res


def testConvergence(image1, image2, tolerance):
    val = norm_2(image1, image2) / norm_1(image1)
    return val <= tolerance
# ----------------------------------------------------------------


def filter_4(image: np.array, lambda_par, tolerance):

    # Parameters
    im = image.copy()
    new_im = im.copy()

    x_range = im.shape[1]
    y_range = im.shape[0]
    w0_par = 1
    const_par = lambda_par / 2
    w1_par = 4 * w0_par - 0 * w0_par
    w2_par = 3 * w0_par - 1 * w0_par
    w3_par = 2 * w0_par - 2 * w0_par
    w4_par = 1 * w0_par - 3 * w0_par
    w5_par = 0 * w0_par - 4 * w0_par
    f1_par = const_par * w1_par
    f2_par = const_par * w2_par
    f3_par = const_par * w3_par
    f4_par = const_par * w4_par
    f5_par = const_par * w5_par

    # Algorithm
    iteration = 0
    has_converged = False
    while not has_converged and iteration <= 26:

        # Asserting new iteration
        iteration += 1
        print("Processing image with lambda %d(%d). Iteration %d" % (lambda_par, tolerance, iteration))
        # Set new array
        im = new_im.copy()

        for y in range(0, y_range):
            for x in range(0, x_range):
                value = im[y][x]
                arr = np.arange(9, dtype=np.dtype(float))
                arr[0] = value if x - 1 < 0 else im[y][x - 1]
                arr[1] = value if x + 1 == x_range else im[y][x + 1]
                arr[2] = value if y - 1 < 0 else im[y - 1][x]
                arr[3] = value if y + 1 == y_range else im[y + 1][x]
                arr[4] = value + f1_par
                arr[5] = value + f2_par
                arr[6] = value + f3_par
                arr[7] = value + f4_par
                arr[8] = value + f5_par
                new_im[y][x] = np.median(arr)

        # Check Convergence
        has_converged = testConvergence(im, new_im, tolerance)

    return new_im

def filter_8(image: np.array, lambda_par, tolerance):

    # Parameters
    im = image.copy()
    new_im = im.copy()

    x_range = im.shape[1]
    y_range = im.shape[0]
    w0_par = 1
    w1_par = 1 / (2 ** (1 / 2))
    u_par = 1 / lambda_par
    const_par = (1 / (2 * u_par))
    w1_val = 4 * w0_par - 0 * w0_par
    w2_val = 4 * w1_par - 0 * w1_par
    w3_val = 3 * w0_par - 1 * w0_par
    w4_val = 3 * w1_par - 1 * w1_par
    w5_val = 2 * w0_par - 2 * w0_par
    w6_val = 1 * w0_par - 3 * w0_par
    w7_val = 1 * w1_par - 3 * w1_par
    w8_val = 0 * w0_par - 4 * w0_par
    w9_val = 0 * w1_par - 4 * w1_par
    f1_val = int(const_par * w1_val)
    f2_val = int(const_par * w2_val)
    f3_val = int(const_par * w3_val)
    f4_val = int(const_par * w4_val)
    f5_val = int(const_par * w5_val)
    f6_val = int(const_par * w6_val)
    f7_val = int(const_par * w7_val)
    f8_val = int(const_par * w8_val)
    f9_val = int(const_par * w9_val)

    # Algorithm
    iter = 0
    has_converged = False
    while not has_converged:

        # Asserting new iteration
        iter += 1
        print("Processing image with lambda %d(%d). Iteration %d" % (lambda_par, tolerance, iter))
        # Set new array
        im = new_im.copy()

        for y in range(0, y_range):
            for x in range(0, x_range):
                value = new_im[y][x]
                arr = np.arange(17)
                arr[0] = value if x - 1 < 0 else new_im[y][x - 1]
                arr[1] = value if x + 1 == x_range else new_im[y][x + 1]
                arr[2] = value if y - 1 < 0 else new_im[y - 1][x]
                arr[3] = value if y + 1 == y_range else new_im[y + 1][x]
                arr[4] = value if (x - 1 < 0 or y - 1 < 0) else new_im[y - 1][x - 1]
                arr[5] = value if (x + 1 == x_range or y - 1 < 0) else new_im[y - 1][x + 1]
                arr[6] = value if (x - 1 < 0 or y + 1 == y_range) else new_im[y + 1][x - 1]
                arr[7] = value if (x + 1 == x_range or y + 1 == y_range) else new_im[y + 1][x + 1]
                arr[8] = min(value + f1_val, 255)
                arr[9] = min(value + f2_val, 255)
                arr[10] = min(value + f3_val, 255)
                arr[11] = min(value + f4_val, 255)
                arr[12] = value + f5_val
                arr[13] = max(value + f6_val, 0)
                arr[14] = max(value + f7_val, 0)
                arr[15] = max(value + f8_val, 0)
                arr[16] = max(value + f9_val, 0)
                new_im[y][x] = np.median(arr)

        # Check Convergence
        has_converged = testConvergence(im, new_im, tolerance)

    return new_im