import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. Đọc và chuẩn bị dữ liệu
# ==========================================
# Đọc tập dữ liệu "X.csv", giả định file không có header
df = pd.read_csv('XuLyDuLieu/X.csv', header=None)
X = df.values

# Lấy số hàng (m) và số cột (n)
m, n = X.shape 

# ==========================================
# 2. Tìm thành phần chính (PCA)
# ==========================================

# Bước a: Quy tâm dữ liệu (Mean Normalization)
# Đưa tâm của X về gốc toạ độ: X = X - mean
mean = np.mean(X, axis=0)
X_norm = X - mean

# Bước b: Tính ma trận phương sai – hiệp phương sai
# Công thức: C = (1/m) * X^T * X
# Dùng np.dot theo yêu cầu
C = (1 / m) * np.dot(X_norm.T, X_norm)

# Bước c: Tìm giá trị riêng (L) và vector riêng (U) của C
L, U = np.linalg.eig(C)

# Sắp xếp các giá trị riêng theo thứ tự giảm dần
index = np.argsort(-L)
L = L[index]
U = U[:, index]

print("Ma trận vector riêng U:\n", U)
print("Các giá trị riêng L:\n", L)

# ==========================================
# 3. Giảm chiều và Phục hồi dữ liệu
# ==========================================

# Tìm 1 thành phần chính của X (k=1), lưu vào Ureduce
k = 1
Ureduce = U[:, 0:k]

# Chiếu dữ liệu gốc lên thành phần chính, lưu kết quả vào Z
# Công thức: Z = X * Ureduce
Z = np.dot(X_norm, Ureduce)

# Phục hồi lại X bằng cách sử dụng Z và Ureduce
# Công thức: X_restore = Z * Ureduce^T + mean
X_approx = np.dot(Z, Ureduce.T)
X_restore = X_approx + mean 

# ==========================================
# 4. Đánh giá sai số
# ==========================================

# Tính sự khác biệt trung bình
diff = X - X_restore
error = (1 / m) * np.sum(diff ** 2)
print(f"\nSự khác biệt trung bình (Reconstruction Error): {error}")

# Tính tổng phương sai
total_variance = (1 / m) * np.sum(X_norm ** 2)
print(f"Tổng phương sai: {total_variance}")

# Tính tỉ lệ sai số
ratio = error / total_variance
print(f"Tỉ lệ (Error / Total Variance): {ratio}")

# ==========================================
# 5. Vẽ đồ thị (Không dùng figure, legend)
# ==========================================

# Vẽ dữ liệu gốc (màu đỏ, chấm tròn)
plt.plot(X[:, 0], X[:, 1], 'ro')

# Vẽ dữ liệu phục hồi (màu xanh, dấu sao)
plt.plot(X_restore[:, 0], X_restore[:, 1], 'b*')

# Vẽ đường nối giữa điểm gốc và điểm phục hồi
for i in range(m):
    plt.plot([X[i, 0], X_restore[i, 0]], [X[i, 1], X_restore[i, 1]], 'k--', alpha=0.3)

plt.title('PCA: Du lieu goc vs Phuc hoi (k=1)')
plt.axis('equal') # Giữ tỉ lệ trục để thấy rõ góc chiếu
plt.grid(True)
plt.show()