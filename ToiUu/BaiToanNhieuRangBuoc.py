from cvxopt import matrix, solvers

# 1. P: Khai báo theo cột.
# Cột 1: [4, 1], Cột 2: [1, 2] -> Ma trận [[4, 1], [1, 2]]
P = matrix([[4.0, 1.0], 
            [1.0, 2.0]])

# 2. q
q = matrix([1.0, 1.0])

# 3. G, h
G = matrix([[-1.0, 0.0], 
            [0.0, -1.0]])
h = matrix([0.0, 0.0])

# Cột 1 là [1.0], Cột 2 là [1.0] -> Tạo thành hàng ngang [1.0, 1.0]
A = matrix([[1.0], [1.0]])
b = matrix([1.0])

# Giải bài toán
sol = solvers.qp(P, q, G, h, A, b)

print("\nKết quả x:")
print(sol['x'])
print("x1 = {:.2f}".format(sol['x'][0]))
print("x2 = {:.2f}".format(sol['x'][1]))