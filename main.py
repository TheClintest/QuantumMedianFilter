from QMF import *
import sys


# ----------------------------------------------------------------
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


# ----------------------------------------------------------------


opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if len(args) != 3:
    raise SystemExit(f'Usage: {sys.argv[0]} [-g] FILENAME LAMBDA EPSILON')

generate_flag = False
mps_flag = None

if "-g" in opts:
    generate_flag = True
if "-mps" in opts:
    mps_flag = 32

# PARAMETERS
color_size = 8
coordinate_size = 2
lambda_par = int(args[1])
epsilon = int(args[2])
optimization = 2

print("###")
print(f"COLORSIZE: {color_size}")
print(f"LAMBDA: {lambda_par}")
print(f"EPSILON: {epsilon}")
print("###")

# INPUT DIRECTORY
input_dir = "./images/"
# OUTPUT DIRECTORY
output_dir = "images/output/"
qasm_dir = "./qasm/"

# IMAGE
filename = f'{args[0]}.png'

# CONVERSION
print(f"Converting image {filename} into array")
img = Converter.to_array(f'{input_dir}{filename}')
patcher = ImagePatcher()
patcher.load_image(img)

# SIMULATOR
print("Setting simulator up")
sim = Simulator(mps_max_bond_dimension=mps_flag)
print(f"Simulator {sim.simulator.name()} is up")

# PRE-TRANSPILING
qmf = QuantumMedianFilter()
if generate_flag:
    print(f'Generating circuits')
    qmf.generate(sim, color_size, coordinate_size, optimization)

# EXECUTION
iteration = 0
converged = False
start_all = time.time()

while not converged or iteration < 24:

    # Prepare iteration
    start_iter = time.time()
    iteration += 1
    pre_img = patcher.get_image()
    post_img = pre_img.copy()
    to_print = ""
    to_iter = f'ITER: {iteration}'

    # Execute for each pixel
    for y in range(1, post_img.shape[0] - 1):
        for x in range(1, post_img.shape[1] - 1):
            to_patch = f'PIXEL: ({y - 1}, {x - 1})'
            to_print = f'{to_iter} {to_patch} '

            # Get neighbors
            neighborhood = dict()
            neighborhood["CNTR"] = post_img[y][x]
            neighborhood["UP"] = post_img[y - 1][x]
            neighborhood["DWN"] = post_img[y + 1][x]
            neighborhood["LFT"] = post_img[y][x - 1]
            neighborhood["RGHT"] = post_img[y][x + 1]

            # Prepare circuit
            qmf.prepare(neighborhood, lambda_par, color_size)
            circuit = qmf.get()

            # Run
            # qobj = sim.transpile(circuit, optimization=0, verbose=True)
            # qobj = load_qasm(f'{qasm_dir}{circuit.name}')
            qobj = circuit
            # print_circuit(circuit, f'{circuit_dir}full.png')
            sys.stdout.write(f"\r{to_print}Simulating the circuit       ")
            answer = sim.simulate(qobj, verbose=False)

            # OUTPUT
            sys.stdout.write(f"\r{to_print}Inserting the result        ")
            out = Converter.decode_pixel(answer, color_size=color_size)
            post_img[y][x] = out

    # End iteration
    sys.stdout.write(f"\r{to_print}Iteration completed        ")
    end_iter = time.time()
    duration_iter = end_iter - start_iter
    print(f'\nITERATION TIME: {duration_iter}')

    # Check convergence
    pre_img = pre_img[1: pre_img.shape[0] - 1, 1: pre_img.shape[1] - 1]
    post_img = post_img[1: post_img.shape[0] - 1, 1: post_img.shape[1] - 1]
    converged = testConvergence(pre_img, post_img, epsilon)

    # Prepare image
    patcher.load_image(post_img)

# Print time
end_all = time.time()
duration_all = end_all - start_all
print(f'\nTOTAL TIME: {duration_all}')

# Save file
output = f'{output_dir}{args[0]}_{lambda_par}_{epsilon}.png'
final = post_img
Converter.to_image(final, filename=output)
print(f'FILE: {output}')
