from PIL import Image
from filtering_4 import *
import numpy as np
import random
import sys
import matplotlib.pyplot as plt


def add_salt_pepper(image: np.array, perc: float):
    x_range = image.shape[1]
    y_range = image.shape[0]
    res = image.copy()
    n = int(x_range * y_range * perc)
    for i in range(n):
        black = random.randint(0, 1)
        x = random.randrange(0, x_range)
        y = random.randrange(0, y_range)
        if black == 1:
            res[y][x] = 0
        else:
            res[y][x] = 255
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

# Preparation
dir = "./images/"
nome = args[0]
file = f"{nome}.png"
salted_file = f'{nome}_sp'
new_file = f'{nome}'

# Loading image and noise addition
image = Image.open(dir + file)
orig_im = np.array(image.convert("L"))

# Parameters
saltpepper = [0.05, 0.1, 0.3]
images_to_filter = dict()
for sp in saltpepper:
    salt_im = add_salt_pepper(orig_im, sp)
    Image.fromarray(salt_im).save(f"{dir}{nome}_sp{int(sp * 100)}.png")
    images_to_filter[sp] = salt_im

# -----------------
tolerance = 0.0001
lambda_par = 1
max_lambda = 255
# -----------------

num_iter = float(127)
temp_iter = max_lambda - lambda_par
temp_iter = float(temp_iter / num_iter)

# Preparing results
results = dict()
temp = lambda_par
while temp <= max_lambda:
    results[temp] = list()
    temp = temp + temp_iter

for label, im in images_to_filter.items():

    lambda_temp = lambda_par
    print("###")
    print(f"Elaborating with error={label}")
    print("###")

    while lambda_temp <= max_lambda:
        # Executing
        new_im = filter_4(im, lambda_temp, tolerance)

        # Saving file
        new_save = np.array(new_im, dtype=np.uint8)
        new_image = Image.fromarray(new_save)
        new_image.save(str(f"%sfiltered/%s_%d_%.1f.png" % (dir, salted_file, label*100, lambda_temp)))

        # Saving RMSE
        rmse_temp = rmse(orig_im, new_im)
        print(f"LAMBDA {lambda_temp} with RMSE {rmse_temp}")
        results[lambda_temp].append(rmse_temp)

        # Refresh
        lambda_temp = lambda_temp + temp_iter

# Plotting results
all_results = dict()
for s in saltpepper:
    all_results[s] = list()
for label, res in results.items():
    for i in range(len(res)):
        all_results[saltpepper[i]].append(res[i])

x_axis = list(results.keys())
for label, res in all_results.items():
    y_axis = res
    min_arg = np.argmin(y_axis)
    xmin = x_axis[min_arg]
    ymin = y_axis[min_arg]
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->")
    kw = dict(xycoords='data', textcoords="offset points", xytext=(0, 40),
              arrowprops=arrowprops, bbox=bbox_props, ha="center", va="bottom")
    ann_text = str("λ=%.1f, RMSE=%.2f" % (xmin, ymin))
    plt.plot(x_axis, y_axis, label=label)
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Filtering results with error = %.0f %%" % (label * 100))
    plt.annotate(ann_text, xy=(xmin, ymin), **kw)
    plt.show()
