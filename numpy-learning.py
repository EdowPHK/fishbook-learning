import numpy as np

# 1
x = np.array([1.0, 2.0, 3.0])
# print(type(x))

# 2
y = np.arange(2, 7, 2)
# print(y)
# print(x * y)

# 3
A = np.array([[1, 2], [3, 4]])
# print(A.shape, A.dtype)

# 4 Broadcasting
B = np.array([5, 6])
# print(A * B)

# 5 
# for row in A:
    # print(row)

# 6
A = A.flatten()
# print(A)
# print(A[np.array([1, 3])])

# 7
# print(A > 2)
# print(A[A > 2])

# 8
x = np.array([-1, 1, 2])
y = x > 0
# print(y.astype(np.int))

# 9
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# print(np.dot(A, B))