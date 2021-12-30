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

# CONVERSION
img = Converter.to_array(f'{input_dir}{filename}')
# PARAMETERS
x_range = img.shape[1]  # X size
y_range = img.shape[0]  # Y size
col_qb = 3  # Size of color register
pos_qb = int(math.ceil(math.log(x_range, 2)))  # Size of position registers
anc_qb = (col_qb - 1) * 2

# QUANTUM REGISTERS
c = QuantumRegister(col_qb, "c")  # Color
x = QuantumRegister(pos_qb, "x")  # X coordinates
y = QuantumRegister(pos_qb, "y")  # Y coordinates
a1 = QuantumRegister(col_qb, "a1_")  # Neighbor 1            |       |
a2 = QuantumRegister(col_qb, "a2_")  # Neighbor 2        1   |   2   |   3
a3 = QuantumRegister(col_qb, "a3_")  # Neighbor 3    _   _   |   _   |   _   _
a4 = QuantumRegister(col_qb, "a4_")  # Neighbor 4            |       |
a5 = QuantumRegister(col_qb, "a5_")  # Neighbor 5        4   |   5   |   6
a6 = QuantumRegister(col_qb, "a6_")  # Neighbor 6    _   _   |   _   |   _   _
a7 = QuantumRegister(col_qb, "a7_")  # Neighbor 7            |       |
a8 = QuantumRegister(col_qb, "a8_")  # Neighbor 8        7   |   8   |   9
a9 = QuantumRegister(col_qb, "a9_")  # Neighbor 9            |       |
e = AncillaRegister(2, "e")
anc = AncillaRegister(anc_qb, "anc")
# CLASSICAL REGISTERS
cm = ClassicalRegister(col_qb, "cm")  # Color Measurement (2)
xm = ClassicalRegister(pos_qb, "xm")  # X Measurement (1)
ym = ClassicalRegister(pos_qb, "ym")  # Y Measurement (0)

# MAIN CIRCUIT
# circuit = QuantumCircuit(c, y, x, a1, a2, a3, a4, a5, a6, a7, a8, a9, cm, ym, xm, name="QMF")
circuit = QuantumCircuit(c, y, x, a1, a2, a3, a4, a5, e, anc, cm, ym, xm, name="QMF4x4")
# CIRCUITS
# sort = Circuit.sort(col_qb, a1.name, a2.name, a3.name)
# prep = Circuit.neighborhood_prep(img, col_qb, verbose=True)
prep_low = Circuit.neighborhood_prep_less(img, col_qb, verbose=True)
# neqr = Circuit.neqr(img, color_num=col_qb, verbose=True)
# cs_w = Circuit.cycleshift_w(pos_qb)
# cs_a = Circuit.cycleshift_a(pos_qb)
# cs_s = Circuit.cycleshift_s(pos_qb)
# cs_d = Circuit.cycleshift_d(pos_qb)
# s = Circuit.swap(col_qb)
# comp = Circuit.comparator(col_qb, a1.name, a2.name, e.name)
# swppr = Circuit.swapper(col_qb)
mmm = Circuit.min_med_max_5(col_qb)

# COMPOSITING
circuit.compose(prep_low, qunion(c, y, x, a1, a2, a3, a4, a5), inplace=True)
circuit.barrier()
circuit.compose(mmm, qunion(a1, a2, a3, a4, a5, e, anc), inplace=True)

# MEASUREMENT
circuit.measure(a3, cm)
circuit.measure(y, ym)
circuit.measure(x, xm)
# VISUALIZATION
style = {
    'displaycolor': {
        "NEQR": "#FF33FF",
        "CS+": "#FF0000",
        "CS-": "#FF8888",
        "SWAP": "#AAAAFF"
    },
    'fontsize': 8
}
# circuit = circuit.decompose() x 7
print(f"QUBITS: {circuit.num_qubits}")
print(f"DEPTH: {circuit.depth()}")
#circuit.draw(output="mpl", reverse_bits=False, initial_state=False, style=style, fold=700)
#plt.savefig(f'{circuit_dir}full_less.png')
#sim = StatevectorSimulator(precision="single", max_parallel_experiments=0, max_parallel_threads =4)
sim = AerSimulator(method="matrix_product_state", matrix_product_state_max_bond_dimension=16)
#c = transpile(circuit, sim, optimization_level=3)
#print(f"COUNT OPS: {c.count_ops()}")
#print(f"DEPTH OPT: {c.depth()}")
# RUN
run = False
t1 = time.time()
if run:
    print("Quantifico!")
    shots = 512
    #sim = AerSimulator(method="statevector")
    qobj = transpile(circuit, sim)
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
        val = val << (8 - col_qb)
        img[y_coord][x_coord] = val
        print("Inserting %d at x:%d y:%d" % (val, x_coord, y_coord))
    Converter.to_image(img, f'{output_dir}output.png')

plt.show()
