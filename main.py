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
images["CHAPLIN"] = "chaplin_64.png"
filename = images["TEST_4x4"]  # Change This One
color_size = 3
# CONVERSION
img = Converter.to_array(f'{input_dir}{filename}')
# CIRCUIT
qmf = QuantumMedianFilter(img, color_size)
# VISUALIZATION
print_circuit(qmf.circuit)
# RUN
sim = AerSimulator(method="matrix_product_state", matrix_product_state_max_bond_dimension=16)
c = transpile(qmf.circuit, sim, optimization_level=3)
run = False
t1 = time.time()
if run:
    print("Quantifico!")
    shots = 512
    #sim = AerSimulator(method="statevector")
    qobj = transpile(c, sim)
    results = sim.run(qobj, shots=shots).result()
    answer = results.get_counts()
    t2 = time.time()
    total =t2 - t1
    print(f"TOTAL TIME:{total}")
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
