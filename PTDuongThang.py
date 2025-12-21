#Tìm 2 đường hỗ trợ của tập dữ liệu âm, dương
from cvxopt import matrix, solvers

P = matrix([[2.0, 0, 0],
           [0, 2, 0],
           [0, 0, 0]])

q = matrix([0.0, 0, 0])

G = matrix([[-2.0, -4, 2],
            [-1, -3, 3],
            [-1, -1, 1]])

h = matrix([-1.0, -1, -1])
ans = solvers.qp(P, q, G, h)
print(ans['x'])