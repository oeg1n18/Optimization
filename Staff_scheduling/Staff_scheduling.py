import cvxpy as cp

x1 = cp.Variable(integer=True)
x2 = cp.Variable(integer=True)
x3 = cp.Variable(integer=True)
x4 = cp.Variable(integer=True)
x5 = cp.Variable(integer=True)
x6 = cp.Variable(integer=True)
x7 = cp.Variable(integer=True)

obj = cp.Minimize((84 * x1 + 84 * x2 + 30 * x3 + 30 * x4 + 30 * x5 + 30 * x6 + 30 * x7))

# Constraints
constraints = [x1 + x2 >= 4,
               x1 + x3 >= 6,
               x1 + x2 + x3 + x4 >= 5,
               x1 + x2 + x3 + x4 + x5 >= 7,
               x2 + x3 + x4 + x5 + x6 >= 8,
               x1 + x4 + x5 + x6 + x7 >= 8,
               x1 + x2 + x5 + x6 + x7 >= 7,
               x1 + x2 + x6 + x7 >= 5,
               x2 + x7 >= 6]


prob = cp.Problem(obj, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x1.value, x2.value, x3.value, x4.value, x5.value, x6.value, x7.value)
