from QMF import *
import sys

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
circuit_dir = "./images/circuits/"
qasm_dir = "./qasm/"

# IMAGE
images = dict()
filename = args[0]

# CONVERSION
print(f"Converting image {filename} into array")
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
sim = Simulator(mps_max_bond_dimension=mps_flag)
print(f"Simulator {sim.simulator.name()} is up")

# PRE-TRANSPILING
qmf = QuantumMedianFilter()
if generate_flag:
    print(f'Generating circuits')
    qmf.generate(sim, color_size, coordinate_size, optimization)

for name, circ in qmf.loaded_circuits.items():
    pass
    # print_circuit(circ, f'{circuit_dir}{name}.png')

# EXECUTION
iteration = 0
start = time.time()
while list(converged_patches.values()).count(False) != 0:

    iteration += 1
    to_print = ""
    to_iter = f'ITER: {iteration}'

    for pos, patch in patches.items():

        to_patch = f'PATCH: {pos}'
        to_print = f'{to_iter} {to_patch} '

        # CIRCUIT
        sys.stdout.write(f"\r{to_print}Building the circuit     ")
        neqr = Circuit.neqr(patch, color_num=color_size, verbose=False)
        neqr_transpiled = sim.transpile(neqr, optimization=0, verbose=False)
        qmf.prepare_old(np.array(patch), lambda_par, color_size, neqr_transpiled)
        circuit = qmf.get()

        # RUN
        # qobj = sim.transpile(circuit, optimization=0, verbose=True)
        # qobj = load_qasm(f'{qasm_dir}{circuit.name}')
        qobj = circuit
        # print_circuit(circuit, f'{circuit_dir}full.png')
        sys.stdout.write(f"\r{to_print}Simulating the circuit       ")
        answer = sim.simulate_old(qobj, shots=128, verbose=False)

        # OUTPUT
        sys.stdout.write(f"\r{to_print}Saving the result        ")
        out = patch.copy()
        out = Converter.decode_image(answer, out, color_size=color_size)
        # Converter.to_image(out, filename=f'{output_dir}patch_{pos[0]}{pos[1]}_{iteration}.png')
        res[pos] = out

    converged_patches = patcher.converged_patches(patches, res, epsilon)
    new = patcher.convert_patches(res)
    output = f'{output_dir}output_{iteration}.png'
    # Converter.to_image(new, filename=output)
    patcher.load_image(new)
    patches = patcher.get_patches()
    res = patches.copy()  # useless?

end = time.time()
total_time = end - start
output = f'{output_dir}output_{lambda_par}_{epsilon}.png'
final = patcher.convert_patches(res)
Converter.to_image(final, filename=output)
print(f'\nTOTAL TIME: {total_time}')
print(f'FILE: {output}')
