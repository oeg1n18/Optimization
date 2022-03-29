import numpy as np
import copy as copy

from LinProgramming import LinProg_2Phase


A1 = np.array([[2, -3]])
b1 = np.array([6])
A2 = np.array([[4 , 5]])
b2 = np.array([20])
C = np.array([3, 1, 0])

test = LinProg_2Phase(C, A1, b1, A2, b2)
test.solve()
