import math
import re
import time

import numpy as np

# importing Qiskit
from qiskit import *
from qiskit.quantum_info import *
from qiskit.providers.aer import *

# import basic plot tools
import matplotlib.pyplot as plt
from sympy.codegen.cfunctions import fma

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
filename = images["TEST_8x8"]  # Change This One
color_size = 3
# CONVERSION
print("Converting image into array")
img = Converter.to_array(f'{input_dir}{filename}')
# CIRCUIT
print("Building the circuit")
qmf = QuantumMedianFilter(img, color_size)
# VISUALIZATION
print("Visualization of the circuit")
print_circuit(qmf.circuit)
# RUN
run = True
if run:
    print("Setting simulator up")
    sim = AerSimulator(method="matrix_product_state", matrix_product_state_max_bond_dimension=32)
    qobj = transpile(qmf.circuit, sim, optimization_level=3)
    print("Running")
    t1 = time.time()
    shots = 2048
    results = sim.run(qobj, shots=shots).result()
    answer = results.get_counts()
    t2 = time.time()
    total =t2 - t1
    print("---RESULTS---")
    print("")
    print(f"Time:{total}")
    print(f"Integrity:{len(answer)}")
    print(f"Integrity:{len(answer)}")
    print(answer)
    for measure in answer:
        m = re.compile(r'\W+').split(measure)
        x_coord = int(m[0], 2)
        y_coord = int(m[1], 2)
        val = int(m[2], 2)
        val = val << (8 - color_size)
        img[y_coord][x_coord] = val
        print("Inserting %d at x:%d y:%d" % (val, x_coord, y_coord))
    Converter.to_image(img, f'{output_dir}output.png')

plt.show()
