from QMF import *

# PARAMETERS
color_size = 4
coordinate_size = 2
lambda_par = 16
epsilon = 16
optimization = 0

# INPUT DIRECTORY
input_dir = "./images/"
# OUTPUT DIRECTORY
output_dir = "./images/test/"
circuit_dir = "./images/circuits/"
qasm_dir = "./qasm/"

# IMAGE
images = dict()
images["TEST_2x2"] = "gray_2.png"
images["TEST_4x4"] = "gray_4.png"
images["TEST_8x8"] = "gray_8.png"
images["GRAY_8x8"] = "gray_shade_8.png"
images["CHAPLIN"] = "chaplin_64.png"
filename = images["TEST_8x8"]  # Change This One

# CONVERSION
print("Converting image into array")
img = Converter.to_array(f'{input_dir}{filename}')
patcher = ImagePatcher()
patcher.load_image(img)
patches = patcher.get_patches()
res = patches.copy()
converged_patches = dict()
for pos in patches.keys():
    converged_patches[pos] = False

# SIMULATOR
print("Setting simulator up")
sim = Simulator(mps_max_bond_dimension=None)

# PRE-TRANSPILING
qmf = QuantumMedianFilter()
# qmf.generate(sim, color_size, coordinate_size, optimization)

# EXECUTION
iteration = 0
start = time.time()
while list(converged_patches.values()).count(False) != 0:

    iteration += 1

    for pos, patch in patches.items():

        print(f'ITER: {iteration}')
        print(f'PATCH: {pos}')

        # CIRCUIT
        print("Building the circuit")
        neqr = Circuit.neqr(patch, color_num=color_size, verbose=False)
        neqr_transpiled = sim.transpile(neqr, optimization=0, verbose=True)
        qmf.prepare(np.array(patch), lambda_par, color_size, neqr_transpiled)
        circuit = qmf.get()

        # RUN
        # qobj = sim.transpile(circuit, optimization=0, verbose=True)
        # qobj = load_qasm(f'{qasm_dir}{circuit.name}')
        qobj = circuit
        answer = sim.simulate(qobj, shots=128, verbose=True)

        # OUTPUT
        out = patch.copy()
        out = Converter.decode_image(answer, out, color_size=color_size)
        # Converter.to_image(out, filename=f'{output_dir}patch_{pos[0]}{pos[1]}_{iteration}.png')
        res[pos] = out

    converged_patches = patcher.converged_patches(patches, res, epsilon)
    new = patcher.convert_patches(res)
    output = f'{output_dir}output_{iteration}.png'
    Converter.to_image(new, filename=output)
    patcher.load_image(new)
    patches = patcher.get_patches()
    res = patches.copy()  # useless?

end = time.time()
total_time = end - start
output = f'{output_dir}output.png'
final = patcher.convert_patches(res)
Converter.to_image(final, filename=output)
print(f'TOTAL TIME: {total_time}')
print(f'FILE: {output}')
