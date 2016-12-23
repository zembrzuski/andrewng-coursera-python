from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

input_matrix = loadtxt('input/ex1data1.txt', comments="#", delimiter=",", unpack=False)
x = input_matrix[:,0]
y = input_matrix[:,1]

# numpy implementation
#fit = [1.17976488, -3.76370194]

# tensorflow implementation
fit = [1.20941746, -3.90434694]

fit_fn = np.poly1d(fit)

plt.plot(x,y, 'yo', x, fit_fn(x), '--k')plt.xlim(0, 25)
plt.ylim(0, 25)
plt.show()

