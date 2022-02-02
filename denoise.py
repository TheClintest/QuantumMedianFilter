import numpy
from threading import Thread
import itertools

from QMF import *
import sys


def generate_single_circuit(pos, patch, pixel, lambda_par, quantumfilter: QuantumMedianFilter, results: dict):
    quantumfilter.prepare_patch(patch, pixel, lambda_par)
    results[pos] = quantumfilter.get()


def generate_circuits(patches: dict, image, lambda_par, quantumfilter: QuantumMedianFilter):
    results = dict()
    threads = list()
    for pos, patch in patches.items():
        y = pos[0] + 1
        x = pos[1] + 1
        pixel = image[y][x]
        threads.append(Thread(target=generate_single_circuit, args=(pos, patch, pixel, lambda_par, quantumfilter, results)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ----------------------------------------------------------------
def norm_1(img: numpy.array):
    res = 0
    x_range = img.shape[1]
    y_range = img.shape[0]
    for x in range(0, x_range):
        for y in range(0, y_range):
            res += int(img[y][x]) ** 2
    return res


def norm_2(img: numpy.array, img_old: numpy.array):
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

def denoise(original_image: np.array, mps_flag, generate_flag, color_size, coordinate_size, lambda_par, tolerance,
            optimization):
    # SIMULATOR
    print("Setting simulator up")
    sim = Simulator(mps_max_bond_dimension=mps_flag)
    print(f"Simulator {sim.simulator.name()} is up")

    # PRE-TRANSPILING
    qmf = QuantumMedianFilter()
    if generate_flag:
        print(f'Generating pre-assembled circuits')
        qmf.generate(sim, color_size, coordinate_size, optimization)

    # PATCHING
    patcher = ImagePatcher()
    patcher.load_image(original_image)

    # EXECUTION
    iteration = 0
    converged = False
    start_all = time.time()

    while not converged and iteration < 24:

        # Prepare iteration
        start_iter = time.time()
        iteration += 1
        pre_img = patcher.get_image()  # Original
        post_img = pre_img.copy()  # New
        all_patches = patcher.get_patches()
        num_concurrent_threads = 384
        num_patches = len(all_patches)
        num_done = 0

        to_print = ""
        to_iter = f'ITER: {iteration}'
        while num_done < num_patches:

            # Slicing patches dictionary
            start = num_done
            stop = num_done + num_concurrent_threads
            patches = dict(itertools.islice(all_patches.items(), start, min(stop, num_patches)))

            # Generate circuit for each pixel
            sys.stdout.write(f"\r{to_iter} Generating circuits  {min(stop, num_patches)}/{num_patches}    ")
            circuits_dict = generate_circuits(patches, pre_img, lambda_par, qmf)

            circuits = list()
            for pos, patch in patches.items():
                circuits.append(circuits_dict[pos])

            # Simulating circuits
            sys.stdout.write(f"\r{to_iter} Simulating circuits  {min(stop, num_patches)}/{num_patches}    ")
            answer = sim.simulate(circuits, verbose=False)

            # Getting results
            sys.stdout.write(f"\r{to_iter} Getting results      {min(stop, num_patches)}/{num_patches}    ")
            ix = 0
            for pos, patch in patches.items():
                res = answer[ix]
                y = pos[0] + 1
                x = pos[1] + 1
                val = Converter.decode_pixel(res)
                post_img[y][x] = val
                ix += 1

            # Refreshing for-cycle
            num_done += num_concurrent_threads


        # End iteration
        sys.stdout.write(f"\r{to_iter} Iteration completed      ")
        end_iter = time.time()
        duration_iter = end_iter - start_iter
        print(f'\nITERATION TIME: {duration_iter}')

        # Check convergence
        pre_img = pre_img[1: pre_img.shape[0] - 1, 1: pre_img.shape[1] - 1]
        post_img = post_img[1: post_img.shape[0] - 1, 1: post_img.shape[1] - 1]
        converged = testConvergence(pre_img, post_img, tolerance)

        # Prepare image
        patcher.load_image(post_img)

    # Print time
    end_all = time.time()
    duration_all = end_all - start_all
    print(f'\nTOTAL TIME: {duration_all}')

    # Save file
    return post_img
