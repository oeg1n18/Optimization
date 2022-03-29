
import cvxpy as cp


# Create two scalar optimization variables.
x1 = cp.Variable(boolean=True)
x2 = cp.Variable(boolean=True)
x3 = cp.Variable(boolean=True)
x4 = cp.Variable(boolean=True)

# Create two constraints.
constraints = [5*x1 + 7*x2 + 4*x3 +3*x4 <= 14]


# Form objective.
obj = cp.Maximize((8*x1 + 11*x2 + 6*x3 + 4*x4))

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x1.value, x2.value, x3.value, x4.value)
