from QMF import *
import sys

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if len(args) != 3:
    raise SystemExit(f'Usage: {sys.argv[0]} [-g] COLORSIZE LAMBDA EPSILON')

generate_flag = False
mps_flag = None

if "-g" in opts:
    generate_flag = True
if "-mps" in opts:
    mps_flag = 24



# PARAMETERS
color_size = int(args[0])
coordinate_size = 2
lambda_par = int(args[1])
epsilon = int(args[2])
optimization = 1

print("###")
print(f"COLORSIZE: {color_size}")
print(f"LAMBDA: {lambda_par}")
print(f"EPSILON: {epsilon}")
print("###")

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
    qmf.generate(sim, color_size, coordinate_size, optimization)

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
        print("#---CIRCUIT INFO---")
        print(f"Qubits: {circuit.qubits.count()}")
        print("#------------------")
        answer = sim.simulate(qobj, shots=64, verbose=True)

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
