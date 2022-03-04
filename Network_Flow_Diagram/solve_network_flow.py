
from scipy.optimize import linprog
import numpy as np

c1 = np.array([4, 4, 2, 2, 6, 1, 2, 2, 1])

A_eq1 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [-1, 0, 1, 1, 1, 0, 0, 0, 0],
                 [0, -1, -1, 0, 0, 1, 1, 0, -1],
                 [0, 0, 0, -1, 0, -1, 0, 1, 0],
                 [0, 0, 0, 0, -1, 0, -1, -1, 1]])

b_eq1 = np.array([20, 0, 0, -5, -15])

# bounds
b_ub1 = np.array([15, 8, 4, 10, 15, 5, 4])

A_ub1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]])

bounds1 = (0, None)


res = linprog(c1, A_ub=A_ub1, b_ub=b_ub1, A_eq=A_eq1, b_eq=b_eq1, method='revised simplex')
print(res)
