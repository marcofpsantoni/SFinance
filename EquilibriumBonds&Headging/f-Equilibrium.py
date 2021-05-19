import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import time

#ne.set_num_threads(4)
#z = 'log(x)/3+2*log(y)/3'
#Z = ne.evaluate(z)


def z(x, y):
    return np.log(x)/3+2*np.log(y)/3


z3d = np.vectorize(z)
x = np.arange(0.1, 2.0, 0.02)
y = np.arange(0.1, 3.0, 0.02)
X, Y = np.meshgrid(x, y)
Z = z3d(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 200)
ax.set_xlabel('Com x')
ax.set_ylabel('Com y')
ax.set_zlabel('Welfare')

plt.show()
