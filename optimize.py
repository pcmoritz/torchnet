from scipy.optimize import minimize
import numpy
import pylab
import struct
from subprocess import call, check_output

def draw_char(img):
    pylab.imshow(numpy.reshape(img, (32, 32)),
                 cmap=pylab.get_cmap("binary"), interpolation='none')

def show_char(img):
    draw_char(img)
    pylab.show()

img = numpy.random.randn(1024)

image = []

for i in range(0, 5):
    infile = open("img-" + str(i+1), "rb")
    image.append(struct.unpack('f'*32*32, infile.read()))
    infile.close()

def compute(X):
    data = list(X)
    out = open("digit" ,"wb")
    s = struct.pack('f'*len(data), *data)
    out.write(s)
    out.close()
    return check_output(["luajit", "optimize.lua"])

def func(X, C=0.0):
    """ Objective function """
    string = compute(X)
    return -float(string.splitlines()[0]) + C*numpy.linalg.norm(X - img)**2

def func_deriv(X, C=0.0):
    """ Derivative of objective function """
    compute(X)
    infile = open("result", "rb")
    s = struct.unpack('f'*32*32, infile.read())
    infile.close()
    grad = numpy.array(s)
    return -grad + 2*C*(X - img)

init = numpy.random.randn(1024)

direction = numpy.random.randn(1024)

def calc_min(init, C):
    return minimize(func, init, args = (C,), jac=func_deriv,
                    method='L-BFGS-B', options={'disp': True, 'maxiter': 150},
                    tol=7e-6,
                    bounds = 1024*[(0, 1)])

def sol_path(l=3,k=1):
    val = numpy.random.randn(1024)
    i = 1
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        res = calc_min(val, C)
        val = res.x
        ax = pylab.subplot(l,6,6*(k-1)+i)
        ax.set_xticks([]) 
        ax.set_yticks([])
        pylab.xlabel(str(func(res.x)))
        i = i + 1
        draw_char(val)

img = image[0]
sol_path(l=5, k=1)
img = image[1]
sol_path(l=5, k=2)
img = image[2]
sol_path(l=5, k=3)
img = image[3]
sol_path(l=5, k=4)
img = image[4]
sol_path(l=5, k=5)
pylab.show()





    
    
    
    
