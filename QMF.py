# IMPORTS

from PIL import Image
import math
import time
import re
import numpy as np
from qiskit import *
from qiskit.circuit.library import *
from qiskit.providers.aer import *
import matplotlib.pyplot as plt


# UTILITY
def qunion(*qregs: QuantumRegister):
    """
    This function returns a list of qubits by their own register variables, useful for Qiskit methods
    :param qregs: Input quantum register
    :return: A list of qubits, ordered by given args
    """
    res = []
    for r in qregs:
        res += r._bits
    return res

# PRINTING
def print_circuit(circuit, filename: str = None):
    """
    Prints out a representation of a given circuit.
    If "filename" is provided, it saves the visualization in the file system.
    :param circuit: Input circuit to visualize
    :param filename: Path for the output
    """
    style = {
        'displaycolor': {
            "NEQR": "#FF33FF",
            "CS+": "#FF0000",
            "CS-": "#FF8888",
            "SWAP": "#AAAAFF"
        },
        'fontsize': 8
    }
    circuit.draw(output="mpl", reverse_bits=False, initial_state=False, style=style, fold=700)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


# QASM
qasm_dir = "./qasm/"


def load_qasm(filename, path='./qasm/', verbose=False):
    """
    Load a QASM string from a path, returning the corresponding circuit
    :param filename: Path to QASM file
    :param verbose: Prints out debug info
    :return: A QuantumCircuit described in the file
    """
    t1 = time.time()
    circ = QuantumCircuit.from_qasm_file(f'{path}{filename}')
    t2 = time.time()
    duration = t2 - t1
    if verbose: print(f'QASM loading time for {filename}: {duration}')
    return circ


def save_qasm(circuit, filename, path='./qasm/', verbose=False):
    """
    Save a circuit/qobj as a QASM file
    :param path: Path to the saving directory
    :param circuit: Target circuit/qobj to save
    :param filename: Name to the file
    :param verbose: Prints out debug info
    """
    t1 = time.time()
    circuit.qasm(filename=f'{path}{filename}')
    t2 = time.time()
    duration = t2 - t1
    if verbose: print(f'QASM saving time: {duration}')


# PATCHER
class ImagePatcher:
    """
    An utility class for image decomposition.
    It generates 2x2 patches of the image, "wrapped" in a 4x4 image.
    Border pixels are repeated to create a sort of frame
    """
    original_image = None
    original_shape = None
    image = None
    max_patch = None
    current_patch = None

    def __preprocess(self):
        """
        An internal method for image padding
        """
        image = self.original_image.copy()
        if self.original_shape[0] % 2 == 1:  # Bottom pad, if necessary
            image = np.pad(image, ((0, 1), (0, 0)), mode="symmetric")
        if self.original_shape[1] % 2 == 1:  # Right pad, if necessary
            image = np.pad(image, ((0, 0), (0, 1)), mode="symmetric")
        self.max_patch = (image.shape[0] // 2, image.shape[1] // 2)
        self.current_patch = (0, 0)
        image = np.pad(image, ((1, 1), (1, 1)), mode="symmetric")  # Total padding
        self.image = image

    def load_image(self, image_array: np.ndarray = None):
        """
        Loads an image array into the patcher and pre-processes it
        :param image_array: A NumPy array encoding the image
        """
        # TODO: Add image check - minimum (2,2)
        self.original_image = image_array
        image_shape = image_array.shape
        self.original_shape = image_shape
        self.__preprocess()

    def get_image(self):
        """
        Returns the original image
        :return: A NumPy array encoding the image
        """
        result = self.image[1:self.original_shape[0] + 1, 1:self.original_shape[1] + 1]
        return result

    def get_patches(self):
        """
        It generates a dictionary containing all possible patches
        :return: A dictionary {pos : patch}
        """
        res = dict()
        for y in range(0, self.max_patch[0]):
            for x in range(0, self.max_patch[1]):
                patch_pos = (y, x)
                x_start = x * 2
                y_start = y * 2
                x_end = x_start + 4
                y_end = y_start + 4
                patch_image = self.image[y_start:y_end, x_start:x_end]
                res[patch_pos] = patch_image
        return res

    def convert_patches(self, patches: dict):
        """
        Converts a patch dictionary into the resulting array, according to image original shape
        :param patches: A dictionary {pos : patch}
        :return: A NumPy array encoding the image
        """
        res = self.image.copy()
        for pos, patch in patches.items():
            y_start = pos[0] * 2 + 1
            x_start = pos[1] * 2 + 1
            y_end = y_start + 2
            x_end = x_start + 2
            to_replace = patch[1:3, 1:3]
            res[y_start:y_end, x_start:x_end] = to_replace
        res = res[1:self.original_shape[0] + 1, 1:self.original_shape[1] + 1]
        return res

    def check_patch(self, a, b, tolerance):
        for x in range(1, 3):
            for y in range(1, 3):
                val1 = int(a[y][x])
                val2 = int(b[y][x])
                res = abs(val2 - val1)
                if res > tolerance:
                    # print("CONVERGENCE %d/%d"%(res,tolerance))
                    return False
        return True

    def converged_patches(self, old: dict, new: dict, epsilon):
        res = dict()
        positions = old.keys()
        for pos in positions:
            res[pos] = self.check_patch(old[pos], new[pos], epsilon)
        return res


# CONVERTER
class Converter:

    # IMAGE CONVERTER
    @staticmethod
    def to_array(filename):
        """
        Converts a PNG image into a greyscale representation. Returns a NumPy array instance of it
        :param filename: Path of input image
        :return: A NumPy array representing input image
        """
        return np.array(Image.open(filename).convert("L"))

    # ARRAY CONVERTER
    @staticmethod
    def to_image(array, filename):
        """
        Converts a Numpy array into a PNG image and saves it as given filename
        :param array: Input image array
        :param filename: Path of output image
        :return:
        """
        new_image = Image.fromarray(array)
        new_image.save(filename)

    # SIMULATION CONVERTER
    @staticmethod
    def decode_image(answer: dict, target_array, color_size=8):
        """
        Process simulator's results to be decoded into an image
        :param answer: Simulator results
        :param target_array: Array for storing result. Use one of the same shape used for encoding
        :param color_size: Size of color registers. Use the value given for the simulated circuit
        """
        # Processing
        for measure in answer:
            m = re.compile(r'\W+').split(measure)
            x_coord = int(m[0], 2)
            y_coord = int(m[1], 2)
            val = int(m[2], 2)
            val = val << (8 - color_size)
            target_array[y_coord][x_coord] = val
            # print("Inserting %d at x:%d y:%d" % (val, x_coord, y_coord))
        # Output
        return target_array


# SIMULATOR
class Simulator:
    """
    An Aer Simulator for experimentation.
    Default setting is "matrix_product_state"
    """

    simulator = None

    def __init__(self, mps_max_bond_dimension: int = None):
        if mps_max_bond_dimension is not None:
            self.simulator = AerSimulator(method="matrix_product_state",
                                          matrix_product_state_max_bond_dimension=mps_max_bond_dimension,
                                          max_parallel_experiments=0,
                                          max_parallel_threads=48
                                          )
        else:
            self.simulator = AerSimulator(method="statevector",
                                          max_parallel_experiments=0,
                                          max_parallel_threads=48
                                          )

    def transpile(self, circuit: QuantumCircuit, optimization=0, qasm_filename=None, verbose=False):
        """
        Transpile circuit
        :param circuit: Target circuit to optimize
        :param optimization: Optimization level for transpiler (0 to 3)
        :param qasm_filename: If path is given, transpiled qobj will be saved as QASM string on file
        :return: Transpiled circuit
        """
        print(f'Transpiling {circuit.name}')
        t1 = time.time()
        qobj = transpile(circuit, self.simulator, optimization_level=optimization)
        t2 = time.time()
        duration = t2 - t1
        if verbose: print(f'Transpiling time: {duration}')
        if qasm_filename is not None:
            if verbose: print(f'Saving circuit as {qasm_filename}')
            save_qasm(qobj, filename=qasm_filename)
        return qobj

    def simulate(self, circuit: QuantumCircuit, shots=1024, verbose=False):
        """
        Simulate experiment
        :param circuit: A quantum circuit to execute
        :param shots: Number of experiments
        :param verbose: Debug printing
        :return: A dictionary with all results.
        """
        print(f'Simulating qobj {circuit.name}')
        t1 = time.time()
        results = self.simulator.run(circuit, shots=shots).result()
        answer = results.get_counts()
        t2 = time.time()
        total = t2 - t1
        if verbose:
            print("---RESULTS---")
            print(f"Time:{total}")
            print(f"Integrity:{len(answer)}")
            print(answer)
            print("-------------")
        return answer


# CIRCUITS
class Circuit:

    # Setter
    @staticmethod
    def setter(number: int, size: int):
        num = f'{number:08b}'[::-1]
        n = QuantumRegister(size, "n")
        circuit = QuantumCircuit(n, name=f'S{number}')
        for ix in range(size):
            if num[ix] == '1':
                circuit.x(n[ix])
        return circuit

    # NEQR Module
    @staticmethod
    def __next_x(val: int, bit: int):
        """
        FOR INTERNAL USAGE: Find next pixel's coordinates to encode
        :param val:
        :param bit:
        :return:
        """
        if (val % (2 ** bit)) == 0:
            return bit
        else:
            return Circuit.__next_x(val, bit - 1)

    @staticmethod
    def neqr(imageArray, color_num=8, verbose=False):
        """
        Encodes the loaded array into a NEQR circuit
        :param color_num: Size of color register
        :param imageArray: A NumPy Array encoding the input image. To create one, use Converter.to_array(...)
        :param verbose: Prints out encoding phase.
        :return: A QuantumCircuit class, encoding the NEQR image
        """

        # PARAMETERS
        if imageArray.shape[0] != imageArray.shape[1]:
            raise Exception("Image array must be a square matrix")
        size = imageArray.shape[1]
        c_qb = color_num  # Size of color register
        n_qb = int(math.ceil(math.log(size, 2)))  # Size of position registers
        # REGISTERS
        c = QuantumRegister(c_qb, "col")
        x = QuantumRegister(n_qb, "x_coor")
        y = QuantumRegister(n_qb, "y_coor")
        pos = QuantumRegister(bits=qunion(x, y))  # Useful for mcx
        qc = QuantumCircuit(c, y, x, name="NEQR")
        # ENCODING
        # Initialize position registers
        qc.x(x)
        qc.x(y)
        # First barrier
        qc.barrier()
        # Encoding colors
        total = 2 ** (n_qb + n_qb)
        val = 0
        i = 1
        while i <= total:
            to_change = Circuit.__next_x(i, n_qb + n_qb - 1)
            val = val ^ (2 ** to_change)  # XOR to get correct coordinate
            x_index = (val >> 0) & (2 ** n_qb) - 1
            y_index = (val >> n_qb)
            pixel = imageArray[y_index][x_index]
            # --debug--
            if verbose:
                print("Encoding %d at x:%d y:%d" % (pixel, x_index, y_index))
            # Set barrier
            qc.barrier()
            # Set X-Gate
            qc.x(pos[to_change])
            # Set CX-Gate
            for n in range(c_qb):
                new_pixel = pixel >> (8 - color_num)  # THIS IS WHERE COLOR ENCODING CHANGES!
                bit = (new_pixel >> n) & 1
                if bit == 1:
                    qc.mcx(pos, c[n])
            # Increase counter
            i += 1
        # Set last barrier
        qc.barrier()
        # Reset coordinates
        qc.x(x)
        qc.x(y)
        # RETURN
        return qc

    # Cycle-Shift Module
    @staticmethod
    def __counter(size, reg_name='q', add=True, module_name="CNTR"):
        q = QuantumRegister(size, reg_name)
        circuit = QuantumCircuit(q, name=f'{module_name}')
        order = range(size)
        if add:
            order = order.__reversed__()
        for i in order:
            if i == 0:
                circuit.x(i)
            else:
                circuit.mcx([n for n in range(i)], i)
        return circuit

    @staticmethod
    def cycleshift_w(size):
        """
        A CycleShift module. Translates Y coordinate up.
        :param size: Size of coordinate's register
        :return: A QuantumCircuit implementing the module
        """
        return Circuit.__counter(size, 'y_coor', add=False, module_name="CSW")

    @staticmethod
    def cycleshift_a(size):
        """
        A CycleShift module. Translates X coordinate left.
        :param size: Size of coordinate's register
        :return: A QuantumCircuit implementing the module
        """
        return Circuit.__counter(size, 'x_coor', add=False, module_name="CSA")

    @staticmethod
    def cycleshift_s(size):
        """
        A CycleShift module. Translates Y coordinate down.
        :param size: Size of coordinate's register
        :return: A QuantumCircuit implementing the module
        """
        return Circuit.__counter(size, 'y_coor', module_name="CSS")

    @staticmethod
    def cycleshift_d(size):
        """
        A CycleShift module. Translates X coordinate right.
        :param size: Size of coordinate's register
        :return: A QuantumCircuit implementing the module
        """
        return Circuit.__counter(size, 'x_coor', module_name="CSD")

    @staticmethod
    def adder_single():
        """
        A 5 qubit circuit with the following logic:
        q0  =   q0  --  A
        q1  =   q1 XOR (q2 XOR q0) -- B/Sum
        q2  =   q2 XOR ((q2 XOR q0) AND (q2 XOR q1)) --Cin/Cout
        q3  =   0   -- q2 XOR q0
        q4  =   0   -- q2 XOR q1

        It adds up the first two qubits (A and B) and the third (C), putting the SUM on q1 and the Cout on q2
        :return: A QuantumCircuit implementing the module
        """
        circuit = QuantumCircuit(5, name="FA")
        circuit.cx(2, 3)  # q3 = C
        circuit.cx(2, 4)  # q4 = C
        circuit.cx(0, 2)  # q2 = C XOR A
        circuit.cx(1, 3)  # q3 = C XOR B
        circuit.cx(2, 1)  # q1 = B XOR ( C XOR A )
        circuit.ccx(2, 3, 4)  # q4 = C XOR ((C XOR A) AND (C XOR B))
        circuit.reset(2)  # q2 = 0
        circuit.reset(3)  # q3 = 0
        circuit.cx(4, 2)  # q2 = C XOR ((C XOR A) AND (C XOR B))
        circuit.reset(4)  # q4 = 0
        return circuit

    @staticmethod
    def adder(size):
        """
        A full adder mod(2**size)
        :param size: Size of addendum registers
        :return: A QuantumCircuit implementing the module
        """
        a = QuantumRegister(size, name="a")
        b = QuantumRegister(size, name="b")
        c0 = AncillaRegister(1, name="c0")
        c1 = AncillaRegister(1, name="c1")
        c2 = AncillaRegister(1, name="c2")
        circuit = QuantumCircuit(a, b, c0, c1, c2, name=f'ADD')
        add = Circuit.adder_single()
        for i in range(size):
            circuit.compose(add.to_instruction(), qubits=[a[i], b[i], c0[0], c1[0], c2[0]], inplace=True)
        # Cap value to 255 if c0 == 1
        circuit.x(c0)
        for i in range(size):
            circuit.x(b[i])
            circuit.ccx(c0[0], b[i], c1[0])
            circuit.x(c1[0])
            circuit.swap(b[i], c1[0])
            circuit.reset(c1)
        circuit.reset(c0)
        return circuit

    @staticmethod
    def subtractor(size):
        """
        A full subtractor mod(2**size)
        :param size: Size of addendum registers
        :return: A QuantumCircuit implementing the module
        """
        a = QuantumRegister(size, name="a")
        b = QuantumRegister(size, name="b")
        c0 = AncillaRegister(1, name="c0")
        c1 = AncillaRegister(1, name="c1")
        c2 = AncillaRegister(1, name="c2")
        circuit = QuantumCircuit(a, b, c0, c1, c2, name=f'SUB')
        adder = Circuit.adder(size)
        circuit.x(a)
        circuit.compose(adder, qunion(a, b, c0, c1, c2), inplace=True)
        circuit.x(a)
        circuit.x(b)
        return circuit

    # Swap Module
    @staticmethod
    def swap(size, a_name="a", b_name="b"):
        """
        A Swap module. Swaps the value of two given registers
        :param size: Size of registers
        :param a_name: Name of the first register
        :param b_name: Name of the second register
        :return: A QuantumCircuit implementing the module
        """
        a = QuantumRegister(size, a_name)
        b = QuantumRegister(size, b_name)
        circuit = QuantumCircuit(a, b, name=f'SWAP')
        for i in range(size):
            circuit.swap(i, i + size)
        return circuit

    # Parallel Controlled-NOT Module
    @staticmethod
    def parallel_controlled_not(size, a_name="a", b_name="b"):
        """
        A Parallel Controlled-NOT module. Copy the value of a given register into another
        :param size: Size of registers
        :param a_name: Name of the original register
        :param b_name: Name of the target register
        :return: A QuantumCircuit implementing the module
        """
        a = QuantumRegister(size, a_name)
        b = QuantumRegister(size, b_name)
        circuit = QuantumCircuit(a, b, name="PCN")
        for i in range(size):
            circuit.cx(i, i + size)
        return circuit

    @staticmethod
    def pcn(size, a_name="a", b_name="b"):
        """
        A Parallel Controlled-NOT module. Copy the value of a given register into another
        :param size: Size of registers
        :param a_name: Name of the original register
        :param b_name: Name of the target register
        :return: A QuantumCircuit implementing the module
        """
        return Circuit.parallel_controlled_not(size, a_name, b_name)

    # Neighborhood Preparation
    @staticmethod
    def neighborhood_prep(img: np.array, f: dict, loaded_circuits: dict, color_size=8, neqr_circuit=None,
                          verbose=False):
        """
        This module process a given image to be prepared for further processing.
        It actually stores a 3x3 mask of the given image on 9 ancillary registers.
        :param neqr_circuit: If given, will use the given NEQR circuit without having to compose it
        :param img: A NumPy array representing the image
        :param color_size: Size of the color registers
        :param verbose: For debug usage
        :return: A QuantumCircuit implementing the module
        """
        # PARAMETERS
        x_range = img.shape[1]  # X size
        y_range = img.shape[0]  # Y size
        col_qb = color_size  # Size of color register
        pos_qb = int(math.ceil(math.log(x_range, 2)))  # Size of position registers
        f1 = int(abs(f['f1'])) >> color_size
        f2 = int(abs(f['f2'])) >> color_size
        f3 = int(abs(f['f3'])) >> color_size
        f4 = int(abs(f['f4'])) >> color_size
        f5 = int(abs(f['f5'])) >> color_size
        # QUANTUM REGISTERS
        c = QuantumRegister(col_qb, "col")  # Color
        x = QuantumRegister(pos_qb, "x_coor")  # X coordinates
        y = QuantumRegister(pos_qb, "y_coor")  # Y coordinates
        a1 = QuantumRegister(col_qb, "a1")  # Neighbor 1            |       |
        a2 = QuantumRegister(col_qb, "a2")  # Neighbor 2        1   |   2   |   3
        a3 = QuantumRegister(col_qb, "a3")  # Neighbor 3    _   _   |   _   |   _   _
        a4 = QuantumRegister(col_qb, "a4")  # Neighbor 4            |       |
        a5 = QuantumRegister(col_qb, "a5")  # Neighbor 5        4   |   5   |   6
        a6 = QuantumRegister(col_qb, "a6")  # Neighbor 6    _   _   |   _   |   _   _
        a7 = QuantumRegister(col_qb, "a7")  # Neighbor 7            |       |
        a8 = QuantumRegister(col_qb, "a8")  # Neighbor 8        7   |   8   |   9
        a9 = QuantumRegister(col_qb, "a9")  # Neighbor 9            |       |
        # ANCILLA REGISTERS
        anc = AncillaRegister(3, "anc")
        # MAIN CIRCUIT
        circuit = QuantumCircuit(c, y, x, a1, a2, a3, a4, a5, a6, a7, a8, a9, anc, name="NBRHD")
        # CIRCUITS
        if neqr_circuit is None:
            neqr = Circuit.neqr(img, color_num=col_qb, verbose=False)
        else:
            neqr = neqr_circuit
        cs_w = loaded_circuits["CSW"]
        cs_a = loaded_circuits["CSA"]
        cs_s = loaded_circuits["CSS"]
        cs_d = loaded_circuits["CSD"]
        swp = loaded_circuits["SWAP"]
        add = loaded_circuits["ADD"]
        sub = loaded_circuits["SUB"]
        q3 = Circuit.setter(f1, col_qb)
        q1 = Circuit.setter(f2, col_qb)
        q7 = Circuit.setter(f4, col_qb)
        q9 = Circuit.setter(f5, col_qb)
        # COMPOSITING
        circuit.h(y)
        circuit.h(x)
        circuit.compose(q3, a3, inplace=True)
        circuit.compose(q1, a1, inplace=True)
        circuit.compose(q7, a7, inplace=True)
        circuit.compose(q9, a9, inplace=True)
        # 5
        if verbose: print("Preparing Pixel:5")
        circuit.compose(neqr, qunion(c, y, x), inplace=True)
        circuit.compose(swp, qunion(c, a5), inplace=True)
        # 6
        if verbose: print("Preparing Pixel:6")
        circuit.compose(cs_d, qunion(x), inplace=True)
        circuit.compose(neqr, qunion(c, y, x), inplace=True)
        circuit.compose(swp, qunion(c, a6), inplace=True)
        # 3
        if verbose: print("Preparing Pixel:3")
        circuit.compose(cs_w, qunion(y), inplace=True)
        circuit.compose(add, qunion(a5, a3, anc), inplace=True)
        # 2
        if verbose: print("Preparing Pixel:2")
        circuit.compose(cs_a, qunion(x), inplace=True)
        circuit.compose(neqr, qunion(c, y, x), inplace=True)
        circuit.compose(swp, qunion(c, a2), inplace=True)
        # 1
        if verbose: print("Preparing Pixel:1")
        circuit.compose(cs_a, qunion(x), inplace=True)
        circuit.compose(add, qunion(a5, a1, anc), inplace=True)
        # 4
        if verbose: print("Preparing Pixel:4")
        circuit.compose(cs_s, qunion(y), inplace=True)
        circuit.compose(neqr, qunion(c, y, x), inplace=True)
        circuit.compose(swp, qunion(c, a4), inplace=True)
        # 7
        if verbose: print("Preparing Pixel:7")
        circuit.compose(cs_s, qunion(y), inplace=True)
        circuit.compose(sub, qunion(a5, a7, anc), inplace=True)
        # 8
        if verbose: print("Preparing Pixel:8")
        circuit.compose(cs_d, qunion(x), inplace=True)
        circuit.compose(neqr, qunion(c, y, x), inplace=True)
        circuit.compose(swp, qunion(c, a8), inplace=True)
        # 9
        if verbose: print("Preparing Pixel:9")
        circuit.compose(cs_d, qunion(x), inplace=True)
        circuit.compose(sub, qunion(a5, a9, anc), inplace=True)
        # Reset
        if verbose: print("Restoring...")
        circuit.compose(cs_w, qunion(y), inplace=True)
        circuit.compose(cs_a, qunion(x), inplace=True)
        # RETURN
        if verbose: print("Done!")
        return circuit

    # Comparator
    @staticmethod
    def comparator(size, a_name="a", b_name="b", res_name="e"):
        """
        This module compares the binary encoding of two registers and stores the result on a third register.
        The result register can result in:
            00: equal
            01: A greater than B
            10: B greater than A
        :param size: Size of the registers
        :param a_name: Name of the first register
        :param b_name: Name of the second register
        :param res_name: Name of the result register
        :return: A QuantumCircuit implementing the module
        """
        # REGISTERS
        anc_qb = (size - 1) * 2
        a = QuantumRegister(size, a_name)
        b = QuantumRegister(size, b_name)
        anc = AncillaRegister(anc_qb, "anc")
        res = AncillaRegister(2, res_name)
        circuit = QuantumCircuit(a, b, res, anc, name="COMP")
        # COMPOSING
        add = []
        for i in range(size).__reversed__():
            if i != 0:
                # Check a > b
                circuit.x(b[i])
                circuit.mcx([a[i], b[i]] + add, anc[i * 2 - 1])
                circuit.mcx([a[i], b[i]] + add, res[0])
                circuit.x(b[i])
                # Check a < b
                circuit.x(a[i])
                circuit.mcx([a[i], b[i]] + add, anc[i * 2 - 2])
                circuit.mcx([a[i], b[i]] + add, res[1])
                circuit.x(a[i])
                # Prepare ancilla
                circuit.x(anc[i * 2 - 1])
                circuit.x(anc[i * 2 - 2])
                add.append(anc[i * 2 - 1])
                add.append(anc[i * 2 - 2])
            else:
                mcmt = MCMT(XGate(), 2 + len(add), 1)
                # Check a > b
                circuit.x(b[i])
                circuit.mcx([a[i], b[i]] + add, res[0])
                circuit.x(b[i])
                # Check b > a
                circuit.x(a[i])
                circuit.mcx([a[i], b[i]] + add, res[1])
                circuit.x(a[i])
            # circuit.barrier()
        # RETURN
        return circuit

    # Swapper
    @staticmethod
    def swapper(size, a_name="a", b_name="b"):
        """
        A Swapper is a composite module which "swaps" two registers based on their binary encoding:
            A > B   --> SWAP
            A <= B  --> PASS
        :param size: Size of the register
        :param a_name: Name of the first register
        :param b_name: Name of the second register
        :return: A QuantumCircuit implementing the module
        """
        # REGISTERS
        anc_qb = (size - 1) * 2
        a = QuantumRegister(size, a_name)
        b = QuantumRegister(size, b_name)
        res = AncillaRegister(2, "e")
        anc = AncillaRegister(anc_qb, "anc")
        circuit = QuantumCircuit(a, b, res, anc, name="SWPPR")
        # COMPOSING
        comp = Circuit.comparator(size, a_name, b_name, res.name)
        swp = Circuit.swap(size, a_name, b_name).control(2, ctrl_state='01')
        circuit.append(comp.to_instruction(), circuit.qubits)
        circuit.append(swp, qunion(res, a, b))
        # RETURN
        return circuit

    # Sort Module
    @staticmethod
    def sort(size, a_name="a", b_name="b", c_name="c"):
        """
        This module sorts a set of three registers.
        :param size: Size of the registers
        :param a_name: Name of the first register
        :param b_name: Name of the second register
        :param c_name: Name of the third register
        :return: A QuantumCircuit implementing the module
        """
        # REGISTERS
        anc_qb = (size - 1) * 2
        a = QuantumRegister(size, a_name)
        b = QuantumRegister(size, b_name)
        c = QuantumRegister(size, c_name)
        res = AncillaRegister(2, "e")
        anc = AncillaRegister(anc_qb, "anc")
        circuit = QuantumCircuit(a, b, c, res, anc, name="SORT")
        # COMPOSING
        swppr_ab = Circuit.swapper(size, a_name=a_name, b_name=b_name).to_instruction()
        swppr_ac = Circuit.swapper(size, a_name=a_name, b_name=c_name).to_instruction()
        swppr_bc = Circuit.swapper(size, a_name=b_name, b_name=c_name).to_instruction()
        # Sort A and B
        circuit.append(swppr_ab, qunion(a, b, res, anc))
        circuit.reset(res)
        circuit.reset(anc)
        # Sort A and C
        circuit.append(swppr_ac, qunion(a, c, res, anc))
        circuit.reset(res)
        circuit.reset(anc)
        # Sort B and C
        circuit.append(swppr_bc, qunion(b, c, res, anc))
        circuit.reset(res)
        circuit.reset(anc)
        # RETURN
        return circuit

    # Maximum-Median-Minimum Module
    @staticmethod
    def min_med_max(size):
        """
        A module which sorts its input, which median is finally stored in the center register.
        :param size: Size of the registers
        :return: A QuantumCircuit implementing the module
        """
        # REGISTERS
        anc_qb = (size - 1) * 2
        a1 = QuantumRegister(size, "a1")
        a2 = QuantumRegister(size, "a2")
        a3 = QuantumRegister(size, "a3")
        a4 = QuantumRegister(size, "a4")
        a5 = QuantumRegister(size, "a5")
        a6 = QuantumRegister(size, "a6")
        a7 = QuantumRegister(size, "a7")
        a8 = QuantumRegister(size, "a8")
        a9 = QuantumRegister(size, "a9")
        res1 = AncillaRegister(2, "e1")
        res2 = AncillaRegister(2, "e2")
        res3 = AncillaRegister(2, "e3")
        anc1 = AncillaRegister(anc_qb, "anc1")
        anc2 = AncillaRegister(anc_qb, "anc2")
        anc3 = AncillaRegister(anc_qb, "anc3")
        circuit = QuantumCircuit(a1, a2, a3, a4, a5, a6, a7, a8, a9, res1, res2, res3, anc1, anc2, anc3,
                                 name=f'MMM')
        # COMPOSING
        sort = Circuit.sort(size).to_instruction()
        # Row sort
        circuit.append(sort, qunion(a1, a2, a3, res1, anc1))
        circuit.append(sort, qunion(a4, a5, a6, res2, anc2))
        circuit.append(sort, qunion(a7, a8, a9, res3, anc3))
        # Column sort
        circuit.append(sort, qunion(a1, a4, a7, res1, anc1))
        circuit.append(sort, qunion(a2, a5, a8, res2, anc2))
        circuit.append(sort, qunion(a3, a6, a9, res3, anc3))
        # Right diagonal sort
        circuit.append(sort, qunion(a3, a5, a7, res1, anc1))
        # RETURN
        return circuit

    @staticmethod
    def min_med_max_5(size):
        """
        A module which sorts its input, which median is finally stored in the center register.
        :param size: Size of the registers
        :return: A QuantumCircuit implementing the module
        """
        # REGISTERS
        anc_qb = (size - 1) * 2
        a1 = QuantumRegister(size, "a1")
        a2 = QuantumRegister(size, "a2")
        a3 = QuantumRegister(size, "a3")
        a4 = QuantumRegister(size, "a4")
        a5 = QuantumRegister(size, "a5")
        res = AncillaRegister(2, "e")
        anc = AncillaRegister(anc_qb, "anc")
        circuit = QuantumCircuit(a1, a2, a3, a4, a5, res, anc, name="MMM5")
        # COMPOSING
        sort = Circuit.sort(size).to_instruction()
        # First sort
        circuit.append(sort, qunion(a2, a3, a4, res, anc))
        # Second sort
        circuit.append(sort, qunion(a1, a3, a5, res, anc))
        # Third sort
        circuit.append(sort, qunion(a2, a3, a4, res, anc))
        # RETURN
        return circuit


# QUANTUM MEDIAN FILTER
class QuantumMedianFilter:
    """
    A Quantum Median Filter.
    OUTPUT: X Y C
    X is a binary representation of pixel horizontal coordinate
    Y is a binary representation of pixel vertical coordinate
    C is a binary representation of pixel median value
    """

    circuit = None
    loaded_circuits = dict()

    def generate(self, simulator: Simulator, color_size: int, coordinate_size: int, optimization_level=3):
        # Circuits
        mmm = Circuit.min_med_max(color_size)
        swp = Circuit.swap(color_size)
        cs_w = Circuit.cycleshift_w(coordinate_size)
        cs_a = Circuit.cycleshift_a(coordinate_size)
        cs_s = Circuit.cycleshift_s(coordinate_size)
        cs_d = Circuit.cycleshift_d(coordinate_size)
        add = Circuit.adder(color_size)
        sub = Circuit.subtractor(color_size)
        # Transpile and save
        self.loaded_circuits["MMM"] = simulator.transpile(mmm, optimization=optimization_level, qasm_filename=mmm.name)
        self.loaded_circuits["SWAP"] = simulator.transpile(swp, optimization=optimization_level, qasm_filename=swp.name)
        self.loaded_circuits["CSW"] = simulator.transpile(cs_w, optimization=optimization_level, qasm_filename=cs_w.name)
        self.loaded_circuits["CSA"] = simulator.transpile(cs_a, optimization=optimization_level, qasm_filename=cs_a.name)
        self.loaded_circuits["CSS"] = simulator.transpile(cs_s, optimization=optimization_level, qasm_filename=cs_s.name)
        self.loaded_circuits["CSD"] = simulator.transpile(cs_d, optimization=optimization_level, qasm_filename=cs_d.name)
        self.loaded_circuits["ADD"] = simulator.transpile(add, optimization=optimization_level, qasm_filename=add.name)
        self.loaded_circuits["SUB"] = simulator.transpile(sub, optimization=optimization_level, qasm_filename=sub.name)

    def load_precompiled_circuits(self):
        to_load = ["MMM", "SWAP", "CSW", "CSA", "CSS", "CSD", "ADD", "SUB"]
        for c in to_load:
            print(f"Loading {c} QASM file")
            self.loaded_circuits[c] = load_qasm(c)

    def prepare(self, img: np.array, lambda_par=1, color_size=8, neqr_circuit=None):
        """
        Prepare the circuit
        :param neqr_circuit: If given, this NEQR circuit will be used, avoiding to compose one
        :param lambda_par: Lambda parameter for filtering
        :param img: A NumPy image representation
        :param color_size: Size of the color registers (defult: 8)
        """
        # IMAGE PARAMETERS
        x_range = img.shape[1]  # X size
        y_range = img.shape[0]  # Y size
        # QC PARAMETERS
        col_qb = color_size  # Size of color register
        pos_qb = int(math.ceil(math.log(x_range, 2)))  # Size of position registers
        anc_qb = (col_qb - 1) * 2
        # FILTER PARAMETERS
        w0_par = 1
        u_par = 1 / lambda_par
        const_par = (1 / (2 * u_par))
        w1_par = 4 * w0_par - 0 * w0_par
        w2_par = 3 * w0_par - 1 * w0_par
        w3_par = 2 * w0_par - 2 * w0_par
        w4_par = 1 * w0_par - 3 * w0_par
        w5_par = 0 * w0_par - 4 * w0_par
        f = dict()
        f['f1'] = const_par * w1_par
        f['f2'] = const_par * w2_par
        f['f3'] = const_par * w3_par
        f['f4'] = const_par * w4_par
        f['f5'] = const_par * w5_par
        # QUANTUM REGISTERS
        c = QuantumRegister(col_qb, "col")  # Color
        x_coord = QuantumRegister(pos_qb, "x_coor")  # X coordinates
        y_coord = QuantumRegister(pos_qb, "y_coor")  # Y coordinates
        a1 = QuantumRegister(col_qb, "a1")
        a2 = QuantumRegister(col_qb, "a2")
        a3 = QuantumRegister(col_qb, "a3")
        a4 = QuantumRegister(col_qb, "a4")
        a5 = QuantumRegister(col_qb, "a5")
        a6 = QuantumRegister(col_qb, "a6")
        a7 = QuantumRegister(col_qb, "a7")
        a8 = QuantumRegister(col_qb, "a8")
        a9 = QuantumRegister(col_qb, "a9")
        # ANCILLA REGISTERS
        res1 = AncillaRegister(2, "e1")
        res2 = AncillaRegister(2, "e2")
        res3 = AncillaRegister(2, "e3")
        anc1 = AncillaRegister(anc_qb, "anc1")
        anc2 = AncillaRegister(anc_qb, "anc2")
        anc3 = AncillaRegister(anc_qb, "anc3")
        anc4 = AncillaRegister(3, "anc4")
        # CLASSICAL REGISTERS
        cm = ClassicalRegister(col_qb, "cm")  # Color Measurement (2)
        xm = ClassicalRegister(pos_qb, "xm")  # X Measurement (1)
        ym = ClassicalRegister(pos_qb, "ym")  # Y Measurement (0)
        # MAIN CIRCUIT
        circuit = QuantumCircuit(c, y_coord, x_coord, a1, a2, a3, a4, a5, a6, a7, a8, a9,  # QUANTUM REGISTERS
                                 res1, res2, res3, anc1, anc2, anc3, anc4,  # ANCILLA REGISTERS
                                 cm, ym, xm,  # CLASSICAL REGISTERS
                                 name="QuantumMedianFilter"  # NAME
                                 )
        # CIRCUITS
        if len(self.loaded_circuits) == 0:
            print("Loading transpiled circuits")
            self.load_precompiled_circuits()
        prep = Circuit.neighborhood_prep(img, f, self.loaded_circuits, color_size=color_size,
                                         neqr_circuit=neqr_circuit)
        mmm = self.loaded_circuits["MMM"]
        swp = self.loaded_circuits["SWAP"]
        # COMPOSITING
        circuit.compose(prep, qunion(c, y_coord, x_coord, a1, a2, a3, a4, a5, a6, a7, a8, a9, anc4), inplace=True)
        circuit.barrier()
        circuit.compose(mmm, qunion(a1, a2, a3, a4, a5, a6, a7, a8, a9, res1, res2, res3, anc1, anc2, anc3),
                        inplace=True)
        circuit.barrier()
        circuit.compose(swp, qunion(c, a5), inplace=True)
        circuit.barrier()
        # MEASUREMENT
        circuit.measure(c, cm)
        circuit.measure(y_coord, ym)
        circuit.measure(x_coord, xm)
        #
        self.circuit = circuit

    def get(self):
        """
        If prepared, returns the Quantum Median Filter module
        :return: A QuantumCircuit implementing the module
        """

        return self.circuit
