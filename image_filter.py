from PIL import Image
import numpy as np

def testConvergence(image1, image2, tolerance):
    x_range = image1.shape[0]
    y_range = image1.shape[1]
    for x in range(0, x_range, 2):
        for y in range(0, y_range, 2):
            val1 = int(image1[x, y])
            val2 = int(image2[x, y])
            res = abs(val2 - val1)
            if (res > tolerance):
                #print("CONVERGENCE %d/%d"%(res,tolerance))
                return False
    return True

dir = "./images/"
file = "image_test_64_gray.png"
new_file = "image_test_64_filtered"

image = Image.open(dir+file)
im = np.array(image.convert("L"))
new_im = im.copy()


lambda_par = 1
while lambda_par <=64:

    #Parameters
    x_range = im.shape[0]
    y_range = im.shape[1]
    eps = 10
    w0_par = 1
    u_par = 1/lambda_par
    const_par = (1/(2*u_par))
    w1_par = 4*w0_par - 0*w0_par
    w2_par = 3*w0_par - 1*w0_par
    w3_par = 2*w0_par - 2*w0_par
    w4_par = 1*w0_par - 3*w0_par
    w5_par = 0*w0_par - 4*w0_par
    f1_par = const_par * w1_par
    f2_par = const_par * w2_par
    f3_par = const_par * w3_par
    f4_par = const_par * w4_par
    f5_par = const_par * w5_par

    #Algorithm
    iter = 0
    has_converged = False
    while(not has_converged):

        #Asserting new iteration
        iter = iter + 1
        print("Processing image with lambda %d. Iteration %d"%(lambda_par, iter))
        #Set new array
        im = new_im.copy()

        #White pattern
        for x in range(0, x_range, 1):
            for y in range(x%2, y_range, 2):
                value = im[x, y]
                arr = np.arange(9)
                arr[0] = value if x - 1 < 0 else im[x - 1, y]
                arr[1] = value if x + 1 == x_range else im[x + 1, y]
                arr[2] = value if y - 1 < 0 else im[x, y - 1]
                arr[3] = value if y + 1 == y_range else im[x, y + 1]
                arr[4] = value + f1_par
                arr[5] = value + f2_par
                arr[6] = value + f3_par
                arr[7] = value + f4_par
                arr[8] = value + f5_par
                new_im[x, y] = np.median(arr)


        #Black pattern
        for x in range(0, x_range, 1):
            for y in range((x+1)%2, y_range, 2):
                value = im[x, y]
                arr = np.arange(9)
                arr[0] = value if x - 1 < 0 else new_im[x - 1, y]
                arr[1] = value if x + 1 == x_range else new_im[x + 1, y]
                arr[2] = value if y - 1 < 0 else new_im[x, y - 1]
                arr[3] = value if y + 1 == y_range else new_im[x, y + 1]
                arr[4] = value + f1_par
                arr[5] = value + f2_par
                arr[6] = value + f3_par
                arr[7] = value + f4_par
                arr[8] = value + f5_par
                new_im[x, y] = np.median(arr)

        #Check Convergence
        has_converged = testConvergence(im, new_im, eps)

    new_image = Image.fromarray(new_im)
    new_image.save("%sfiltered/%s_%d.png"%(dir,new_file, lambda_par))
    lambda_par = lambda_par*2



