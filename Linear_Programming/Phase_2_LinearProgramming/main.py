import numpy as np
import copy as copy

from LinProgramming import LinProg_phase2


A = np.array([[2, -3], [4, 5]])
b = np.array([[6], [20]])
C = np.array([[3, 1, 0]])


test = LinProg_phase2(C, A, b)
test.solve()


print(test.solution)
