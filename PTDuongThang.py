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
w1 = ans['x'][0]
w2 = ans['x'][1]
b  = ans['x'][2]

# In ra phương trình
print("-" * 30)
print("KẾT QUẢ CUỐI CÙNG:")
print(f"w1 = {w1:.2f}")
print(f"w2 = {w2:.2f}")
print(f"b  = {b:.8f}") # In nhiều số thập phân để thấy nó gần bằng 0

print("-" * 30)
# Dùng :.2f để làm tròn 2 chữ số thập phân cho gọn
print(f"Phương trình đường phân lớp: {w1:.1f}*x1 + ({w2:.1f})*x2 + {b:.1f} = 0")
print("-" * 30)