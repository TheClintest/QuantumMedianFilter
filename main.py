from QMF import *

# INPUT DIRECTORY
input_dir = "./images/"
# OUTPUT DIRECTORY
output_dir = "./images/test/"
circuit_dir = "./images/circuits/"

# IMAGE
images = dict()
images["TEST_2x2"] = "gray_2.png"
images["TEST_4x4"] = "gray_4.png"
images["TEST_8x8"] = "gray_8.png"
images["CHAPLIN"] = "chaplin_64.png"
filename = images["TEST_4x4"]  # Change This One
color_size = 4
# CONVERSION
print("Converting image into array")
img = Converter.to_array(f'{input_dir}{filename}')
# CIRCUIT
print("Building the circuit")
qmf = QuantumMedianFilter()
qmf.prepare_5(img, color_size)
circuit = qmf.get()
# VISUALIZATION
print("Visualization of the circuit")
print_circuit(circuit)
# RUN
print("Setting simulator up")
sim = Simulator(mps_max_bond_dimension=64)
# qobj = sim.transpile(circuit, optimization=3, qasm_filename="./qasm/circ_test")
qobj = QuantumCircuit.from_qasm_file("./qasm/circ_test")
answer = sim.simulate(qobj, shots=512, verbose=True)
# OUTPUT
output = f'{output_dir}output.png'
out = img.copy()
out = Converter.decode_image(answer, out, color_size=color_size)
Converter.to_image(out, filename=output)
