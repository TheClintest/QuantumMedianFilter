from PIL import Image
import numpy as np


def testConvergence(image1, image2, tolerance):
    x_range = image1.shape[0]
    y_range = image1.shape[1]
    for x in range(0, x_range):
        for y in range(0, y_range):
            val1 = int(image1[y][x])
            val2 = int(image2[y][x])
            res = abs(val2 - val1)
            if (res > tolerance):
                # print("CONVERGENCE %d/%d"%(res,tolerance))
                return False
    return True


dir = "./images/"
file = "gray_8.png"
new_file = "gray_8_filtered"

image = Image.open(dir + file)
eps = 8
lambda_par = 1
while lambda_par <= 256:

    # Parameters
    im = np.array(image.convert("L"))
    new_im = im.copy()
    x_range = im.shape[0]
    y_range = im.shape[1]
    w0_par = 1
    u_par = 1 / lambda_par
    const_par = (1 / (2 * u_par))
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
    iter = 0
    has_converged = False
    while not has_converged:

        # Asserting new iteration
        iter += 1
        print("Processing image with lambda %d(%d). Iteration %d" % (lambda_par, eps, iter))
        # Set new array
        im = new_im.copy()

        for y in range(0, y_range):
            for x in range(0, x_range):
                value = im[y][x]
                arr = np.arange(9)
                arr[0] = value if x - 1 < 0 else im[x - 1, y]
                arr[1] = value if x + 1 == x_range else im[x + 1, y]
                arr[2] = value if y - 1 < 0 else im[x, y - 1]
                arr[3] = value if y + 1 == y_range else im[x, y + 1]
                arr[4] = min(value + f1_par, 255)
                arr[5] = min(value + f2_par, 255)
                arr[6] = value + f3_par
                arr[7] = max(value + f4_par, 0)
                arr[8] = max(value + f5_par, 0)
                new_im[y][x] = np.median(arr)

        # Check Convergence
        has_converged = testConvergence(im, new_im, eps)

    new_image = Image.fromarray(new_im)
    new_image.save("%sfiltered/%s_%d.png" % (dir, new_file, lambda_par))
    lambda_par = lambda_par * 2

