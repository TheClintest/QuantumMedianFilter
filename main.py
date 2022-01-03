from QMF import *

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
images["CHAPLIN"] = "chaplin_64.png"
filename = images["TEST_8x8"]  # Change This One
color_size = 4
coor_size = 3
# CONVERSION
print("Converting image into array")
img = Converter.to_array(f'{input_dir}{filename}')
patcher = ImagePatcher()
patcher.load_image(img)
patches = patcher.get_patches()
res = patches.copy()
for pos, patch in patches.items():

    print(f'PATCH: {pos}')

    # CIRCUIT
    print("Building the circuit")
    qmf = QuantumMedianFilter()
    qmf.prepare_5(np.array(patch), color_size)
    circuit = qmf.get()

    # RUN
    print("Setting simulator up")
    sim = Simulator(mps_max_bond_dimension=32)
    qobj = sim.transpile(circuit, optimization=0, qasm_filename=f'{qasm_dir}{circuit.name}')
    # qobj = load_qasm(f'{qasm_dir}{circuit.name}')
    answer = sim.simulate(qobj, shots=256, verbose=True)

    # OUTPUT
    out = patch.copy()
    out = Converter.decode_image(answer, out, color_size=color_size)
    # Converter.to_image(out, filename=output)
    res[pos] = out

output = f'{output_dir}output.png'
final = patcher.convert_patches(res)
Converter.to_image(final, filename=output)
