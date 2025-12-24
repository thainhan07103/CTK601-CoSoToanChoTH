import numpy as np
from cvxopt import matrix, solvers

# --- BƯỚC 1: KHAI BÁO BẰNG NUMPY (Dễ nhìn, quen thuộc) ---
# Lưu ý quan trọng: Phải ép kiểu về số thực (float) vì CVXOpt không chơi với số nguyên (int)

# P: Ma trận đối xứng (hệ số bậc 2 nhân đôi)
# Viết theo hàng ngang bình thường: [[hàng 1], [hàng 2]]
P_np = np.array([
    [2.0, 0.0],
    [0.0, 4.0]
])

# q: Vector hệ số bậc 1 (nhớ đổi dấu nếu bài toán gốc là Max)
q_np = np.array([-4.0, -4.0])

# A: Ma trận hệ số ràng buộc (1 phương trình, 2 ẩn => 1 hàng, 2 cột)
A_np = np.array([[1.0, 4.0]])

# b: Vế phải của ràng buộc
b_np = np.array([3.0])

# --- BƯỚC 2: CONVERT SANG CVXOPT ---
# Chỉ cần bọc cvxopt.matrix() bên ngoài là xong
P = matrix(P_np)
q = matrix(q_np)
A = matrix(A_np)
b = matrix(b_np)

# --- BƯỚC 3: GIẢI (SOLVE) ---
sol = solvers.qp(P, q, G=None, h=None, A=A, b=b)

# --- BƯỚC 4: LẤY KẾT QUẢ VỀ NUMPY ĐỂ XÀI TIẾP ---
# sol['x'] trả về ma trận CVXOpt, ta chuyển ngược lại về NumPy array cho dễ dùng
result_x = np.array(sol['x'])

print("\n=== KẾT QUẢ ===")
print("Nghiệm x tìm được (dạng NumPy):")
print(result_x)

x1 = result_x[0][0]
x2 = result_x[1][0]

print(f"\nx1 = {x1}")
print(f"x2 = {x2}")

# Tính lại giá trị hàm mục tiêu gốc (Max)
# f(x) = 5 - (x1 - 2)^2 - 2(x2 - 1)^2
max_val = 5 - (x1 - 2)**2 - 2*(x2 - 1)**2
print(f"Giá trị Max của hàm số: {max_val}")