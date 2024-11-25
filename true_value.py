import numpy as np

# Coefficient matrix A
A = np.array([
    [1, -0.45, 0, 0, 0],
    [-0.45, 1, -0.45, 0, 0],
    [0, -0.45, 1, -0.45, 0],
    [0, 0, -0.45, 1, -0.45],
    [0, 0, 0, -0.45, 1]
])

# Constant vector b
b = np.array([0, 0, 0, 0, 0.45])

# Solve the linear equation Ax = b
V = np.linalg.solve(A, b)

# Output the results
for i, v in enumerate(V, start=1):
    print(f"V({i}) = {v:.3f}")
