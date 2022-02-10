from PIL import Image
from filtering_4 import *
import numpy as np
import random
import sys
import matplotlib.pyplot as plt


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


def add_gaussian(image: np.array, n_med, n_sigma):
    res = image.copy()
    x_range = image.shape[1]
    y_range = image.shape[0]

    noise = np.random.randn(y_range, x_range)
    noise = noise * n_sigma
    res = res + noise
    return res


def rmse(start_image, end_image):
    x_range = start_image.shape[1]
    y_range = start_image.shape[0]
    res = norm_2(start_image, end_image)
    res = res / (x_range * y_range)
    res = pow(res, 1 / 2)
    return res


# ----------------------------------------------------------------

# Checking input parameters
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

sp_flag = False
g_flag = False
all_flag = False

if "-sp" in opts:
    sp_flag = True
if "-g" in opts:
    g_flag = True
if "-a" in opts:
    all_flag = True

# Preparation
dir = "./images/"
nome = args[0]
file = f"{nome}.png"
salted_file = f'{nome}_sp'
gaussian_file = f'{nome}_g'
new_file = f'{nome}'

# Loading image and noise addition
image = Image.open(dir + file)
orig_im = np.array(image.convert("L"))
# plt.imshow(im, cmap='gray', vmin=0, vmax=255)
if sp_flag or all_flag:
    salt_im = add_salt_pepper(orig_im, 3500)
    Image.fromarray(salt_im).save("%s%s_sp.png" % (dir, nome))
if g_flag or all_flag:
    gauss_image = add_gaussian(orig_im, 0, 15)
    # plt.imshow(gauss_image, cmap='gray', vmin=0, vmax=255)
    gauss_tosave = np.array(gauss_image, dtype=np.uint8)
    Image.fromarray(gauss_tosave).save("%s%s_g.png" % (dir, nome))

# Parameters
images_to_filter = dict()
if sp_flag or all_flag:
    images_to_filter["SP"] = salt_im.copy()
if g_flag or all_flag:
    images_to_filter["GAUSS"] = gauss_image.copy()
if not sp_flag and not g_flag:
    images_to_filter["NORM"] = orig_im.copy()

# -----------------
tolerance = 0.0001
lambda_par = 2
max_lambda = 512
# -----------------

# Preparing results
results = dict()
temp = lambda_par
while temp <= max_lambda:
    results[str(temp)] = list()
    temp = temp * 2

for label, im in images_to_filter.items():

    lambda_temp = lambda_par
    print("###")
    print(f"Elaborating {label}")
    print("###")

    while lambda_temp <= max_lambda:

        new_im = filter_8(im, lambda_temp, tolerance)

        # Saving file
        if label == "GAUSS":
            new_save = np.array(new_im, dtype=np.uint8)
            new_image = Image.fromarray(new_save)
        else:
            new_image = Image.fromarray(new_im)

        if label == "NORM":
            new_image.save("%sfiltered/%s_%d.png" % (dir, new_file, lambda_temp))
        if label == "SP":
            new_image.save("%sfiltered/%s_%d.png" % (dir, salted_file, lambda_temp))
        if label == "GAUSS":
            new_image.save("%sfiltered/%s_%d.png" % (dir, gaussian_file, lambda_temp))

        # Saving RMSE
        results[str(lambda_temp)].append(rmse(orig_im, np.array(new_im, dtype=np.uint8)))

        # Refresh
        lambda_temp = lambda_temp * 2

# Plotting results
x_axis = list(results.keys())
y_axis = list(results.values())
plt.plot(x_axis, y_axis)
plt.show()