from PIL import Image
import numpy as np
import random


# ----------------------------------------------------------------
def norm_1(img: np.array):
    res = 0
    x_range = img.shape[1]
    y_range = img.shape[0]
    for x in range(0, x_range):
        for y in range(0, y_range):
            res += (int(img[y][x])) ** 2
    return res


def norm_2(img: np.array, img_old: np.array):
    res = 0
    x_range = img.shape[1]
    y_range = img.shape[0]
    for x in range(0, x_range):
        for y in range(0, y_range):
            res += (int(img[y][x]) - int(img_old[y][x])) ** 2
    return res


# ----------------------------------------------------------------
def testConvergence(image1, image2, tolerance):
    val = norm_2(image1, image2) / norm_1(image1)
    return val <= tolerance


# ----------------------------------------------------------------

def add_salt_pepper(image: np.array, n):
    x_range = image.shape[1]
    y_range = image.shape[0]
    res = image.copy()
    for i in range(n):
        black = random.randint(0, 1)
        x = random.randrange(0, x_range)
        y = random.randrange(0, y_range)
        if black == 1:
            res[y][x] = 0
        else:
            res[y][x] = 255
    return res

salt_pepper = True

dir = "./images/"
nome = "lena"
file = f"{nome}.png"
if salt_pepper:
    new_file = f'{nome}_sp'
else:
    new_file = nome

image = Image.open(dir + file)
im = np.array(image.convert("L"))
if salt_pepper:
    salt_im = add_salt_pepper(im, 512)
    Image.fromarray(salt_im).save("%s%s_sp.png" % (dir, nome))
eps = 0.00001
lambda_par = 2
while lambda_par <= 256:

    # Parameters
    if salt_pepper:
        im = salt_im.copy()
    else:
        im = np.array(image.convert("L"))
    new_im = im.copy()

    x_range = im.shape[1]
    y_range = im.shape[0]
    w0_par = 1
    u_par = 1 / lambda_par
    const_par = (1 / (2 * u_par))
    w1_par = 4 * w0_par - 0 * w0_par
    w2_par = 3 * w0_par - 1 * w0_par
    w3_par = 2 * w0_par - 2 * w0_par
    w4_par = 1 * w0_par - 3 * w0_par
    w5_par = 0 * w0_par - 4 * w0_par
    f1_par = int(const_par * w1_par)
    f2_par = int(const_par * w2_par)
    f3_par = int(const_par * w3_par)
    f4_par = int(const_par * w4_par)
    f5_par = int(const_par * w5_par)

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
                arr[0] = value if x - 1 < 0 else im[y][x - 1]
                arr[1] = value if x + 1 == x_range else im[y][x + 1]
                arr[2] = value if y - 1 < 0 else im[y - 1][x]
                arr[3] = value if y + 1 == y_range else im[y + 1][x]
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

