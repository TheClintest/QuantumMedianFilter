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
circuit = qmf.circuit
# VISUALIZATION
print("Visualization of the circuit")
print_circuit(circuit)
# RUN
print("Setting simulator up")
sim = Simulator(mps_max_bond_dimension=16)
answer = sim.simulate(circuit, shots=512, optimization=3, verbose=True)
# OUTPUT
output = f'{output_dir}output.png'
Converter.decode_image(answer, output, color_size=color_size)
