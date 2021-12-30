# IMPORTS
from PIL import Image
import math
import numpy as np
from qiskit import *
from qiskit.circuit.library import *

def qunion(*qregs : QuantumRegister):
    res = []
    for r in qregs:
        res += r._bits
    return res

# CONVERTER
class Converter:

    # IMAGE CONVERTER
    @staticmethod
    def to_array(filename):
        """Converts a PNG image into a greyscale representation. Returns a NumPy array instance of it"""
        return np.array(Image.open(filename).convert("L"))

    # ARRAY CONVERTER
    @staticmethod
    def to_image(array, filename):
        """Converts a Numpy array into a PNG image and saves it as given filename"""
        new_image = Image.fromarray(array)
        new_image.save(filename)


# CIRCUITS
class Circuit:

    # NEQR Module
    @staticmethod
    def __next_x(val: int, bit: int):
        if (val % (2 ** bit)) == 0:
            return bit
        else:
            return Circuit.__next_x(val, bit - 1)

    @staticmethod
    def neqr(imageArray, color_num=8, verbose=False):
        """
                Encodes the loaded array into a NEQR circuit
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
        c = QuantumRegister(c_qb, "c")
        x = QuantumRegister(n_qb, "x")
        y = QuantumRegister(n_qb, "y")
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
                new_pixel = pixel >> (8-color_num)
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
        circuit = QuantumCircuit(q, name=module_name)
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
        return Circuit.__counter(size, 'y', add=False, module_name="CS-")

    @staticmethod
    def cycleshift_a(size):
        return Circuit.__counter(size, 'x', add=False, module_name="CS-")

    @staticmethod
    def cycleshift_s(size):
        return Circuit.__counter(size, 'y', module_name="CS+")

    @staticmethod
    def cycleshift_d(size):
        return Circuit.__counter(size, 'x', module_name="CS+")

    # Swap Module
    @staticmethod
    def swap(size, a_name="a", b_name="b"):
        a = QuantumRegister(size, a_name)
        b = QuantumRegister(size, b_name)
        circuit = QuantumCircuit(a, b, name="SWAP")
        for i in range(size):
            circuit.swap(i, i + size)
        return circuit

    # Parallel Controlled-NOT Module
    @staticmethod
    def parallel_controlled_not(size, a_name="a", b_name="b"):
        a = QuantumRegister(size, a_name)
        b = QuantumRegister(size, b_name)
        circuit = QuantumCircuit(a, b, name="Parallel Controlled-NOT")
        for i in range(size):
            circuit.cx(i, i + size)
        return circuit

    @staticmethod
    def pcn(size, a_name="a", b_name="b"):
        return Circuit.parallel_controlled_not(size, a_name, b_name)

    # Neighborhood Preparation
    @staticmethod
    def neighborhood_prep(img, color_num=8, verbose=False):
        # PARAMETERS
        x_range = img.shape[1]  # X size
        y_range = img.shape[0]  # Y size
        col_qb = color_num  # Size of color register
        pos_qb = int(math.ceil(math.log(x_range, 2)))  # Size of position registers
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
        # MAIN CIRCUIT
        circuit = QuantumCircuit(c, y, x, a1, a2, a3, a4, a5, a6, a7, a8, a9, name="NBRHD")
        # CIRCUITS
        neqr = Circuit.neqr(img, color_num=col_qb, verbose=False).to_instruction()
        cs_w = Circuit.cycleshift_w(pos_qb).to_instruction()
        cs_a = Circuit.cycleshift_a(pos_qb).to_instruction()
        cs_s = Circuit.cycleshift_s(pos_qb).to_instruction()
        cs_d = Circuit.cycleshift_d(pos_qb).to_instruction()
        swp = Circuit.swap(col_qb).to_instruction()
        # COMPOSITING
        circuit.h(y)
        circuit.h(x)
        # 5
        if verbose: print("Preparing Pixel:5")
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a5))
        # 6
        if verbose: print("Preparing Pixel:6")
        circuit.append(cs_d, qunion(x))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a6))
        # 3
        if verbose: print("Preparing Pixel:3")
        circuit.append(cs_w, qunion(y))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a3))
        # 2
        if verbose: print("Preparing Pixel:2")
        circuit.append(cs_a, qunion(x))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a2))
        # 1
        if verbose: print("Preparing Pixel:1")
        circuit.append(cs_a, qunion(x))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a1))
        # 4
        if verbose: print("Preparing Pixel:4")
        circuit.append(cs_s, qunion(y))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a4))
        # 7
        if verbose: print("Preparing Pixel:7")
        circuit.append(cs_s, qunion(y))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a7))
        # 8
        if verbose: print("Preparing Pixel:8")
        circuit.append(cs_d, qunion(x))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a8))
        # 9
        if verbose: print("Preparing Pixel:9")
        circuit.append(cs_d, qunion(x))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a9))
        # Reset
        if verbose : print("Restoring...")
        circuit.append(cs_w, qunion(y))
        circuit.append(cs_a, qunion(x))
        # RETURN
        if verbose : print("Done!")
        return circuit

    @staticmethod
    def neighborhood_prep_less(img, color_num=8, verbose=False):
        # PARAMETERS
        x_range = img.shape[1]  # X size
        y_range = img.shape[0]  # Y size
        col_qb = color_num  # Size of color register
        pos_qb = int(math.ceil(math.log(x_range, 2)))  # Size of position registers
        # QUANTUM REGISTERS
        c = QuantumRegister(col_qb, "c")  # Color
        x = QuantumRegister(pos_qb, "x")  # X coordinates
        y = QuantumRegister(pos_qb, "y")  # Y coordinates
        a1 = QuantumRegister(col_qb, "a1_")  # Neighbor 1
        a2 = QuantumRegister(col_qb, "a2_")  # Neighbor 2       |1|
        a3 = QuantumRegister(col_qb, "a3_")  # Neighbor 3    |2||3||4|
        a4 = QuantumRegister(col_qb, "a4_")  # Neighbor 4       |5|
        a5 = QuantumRegister(col_qb, "a5_")  # Neighbor 5

        # MAIN CIRCUIT
        circuit = QuantumCircuit(c, y, x, a1, a2, a3, a4, a5, name="NBRHD")
        # CIRCUITS
        neqr = Circuit.neqr(img, color_num=col_qb, verbose=False).to_instruction()
        cs_w = Circuit.cycleshift_w(pos_qb).to_instruction()
        cs_a = Circuit.cycleshift_a(pos_qb).to_instruction()
        cs_s = Circuit.cycleshift_s(pos_qb).to_instruction()
        cs_d = Circuit.cycleshift_d(pos_qb).to_instruction()
        swp = Circuit.swap(col_qb)
        # COMPOSITING
        circuit.h(y)
        circuit.h(x)
        # 3
        if verbose: print("Preparing Pixel:3")
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a3))
        # 4
        if verbose: print("Preparing Pixel:4")
        circuit.append(cs_d, qunion(x))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a4))
        # 1
        if verbose: print("Preparing Pixel:1")
        circuit.append(cs_w, qunion(y))
        circuit.append(cs_a, qunion(x))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a1))
        # 2
        if verbose: print("Preparing Pixel:2")
        circuit.append(cs_a, qunion(x))
        circuit.append(cs_s, qunion(y))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a2))
        # 5
        if verbose: print("Preparing Pixel:5")
        circuit.append(cs_s, qunion(y))
        circuit.append(cs_d, qunion(x))
        circuit.append(neqr, qunion(c, y, x))
        circuit.append(swp, qunion(c, a5))
        # Reset
        if verbose: print("Restoring...")
        circuit.append(cs_w, qunion(y))
        # RETURN
        if verbose: print("Done!")
        return circuit

    # Comparator
    @staticmethod
    def comparator(size, a_name="a", b_name="b", res_name="e"):
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
            #circuit.barrier()
        # RETURN
        return circuit

    # Swapper
    @staticmethod
    def swapper(size, a_name="a", b_name="b"):
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
        pass

    @staticmethod
    def min_med_max_5(size):
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
class QuantumMedianFilter(QuantumCircuit):
    pass

# SIMULATOR
