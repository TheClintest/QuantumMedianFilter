import sys
from denoise import *

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if len(args) != 3:
    raise SystemExit(f'Usage: {sys.argv[0]} [-g -mps -d] FILENAME LAMBDA EPSILON')

generate_flag = False
denoise_flag = False
mps_flag = None

if "-g" in opts:
    generate_flag = True
if "-mps" in opts:
    mps_flag = 32
if "-d" in opts:
    denoise_flag = True

# PARAMETERS
color_size = 8
lambda_par = float(args[1])
epsilon = float(args[2])

print("###")
print(f"COLORSIZE: {color_size}")
print(f"LAMBDA: {lambda_par}")
print(f"EPSILON: {epsilon}")
print("###")

# INPUT DIRECTORY
input_dir = "./images/"
# OUTPUT DIRECTORY
output_dir = "./images/output/"
qasm_dir = "./qasm/"

# IMAGE
filename = f'{args[0]}.png'

# CONVERSION
print(f"Converting image {filename} into array")
img = Converter.to_array(f'{input_dir}{filename}')

#EXECUTION
if denoise_flag:
    output = f'{output_dir}{args[0]}_{lambda_par}.png'
    res = denoise(img, mps_flag, generate_flag, color_size, 1, lambda_par, epsilon, optimization=2)
    Converter.to_image(res, filename=output)
    print(f"FILE: {output}")

