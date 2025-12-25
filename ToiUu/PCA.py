import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# BÀI TẬP 4: PHÂN TÍCH THÀNH PHẦN CHÍNH (PCA)
# ==========================================================

# --- YÊU CẦU: Đọc tập dữ liệu "X.csv" lưu vào biến X ---
# (Tập dữ liệu này chỉ có 2 cột, n=2)
df = pd.read_csv('XuLyDuLieu/X.csv', header=None)
X = df.values
m = X.shape[0] # Số lượng mẫu dữ liệu

# --- YÊU CẦU: Vẽ tập dữ liệu X (dùng 'ro') ---
# Vẽ dữ liệu gốc màu đỏ ('ro')
plt.plot(X[:, 0], X[:, 1], 'ro', label='Dữ liệu gốc')

# --- YÊU CẦU: Tìm 1 thành phần chính của X (k=1), lưu vào Ureduce ---
# (Bước này đề bài ẩn các bước phụ, nhưng theo lý thuyết cần làm như sau)

# 1. Quy tâm dữ liệu (Gợi ý: đưa tâm về gốc tọa độ)
mean = np.mean(X, axis=0)
X_center = X - mean

# 2. Tính ma trận hiệp phương sai
# Công thức: C = (1/m) * X^T * X
C = (1/m) * np.dot(X_center.T, X_center)

# 3. Phân tích giá trị riêng và vector riêng
vals, vecs = np.linalg.eig(C)

# 4. Sắp xếp giảm dần theo giá trị riêng để chọn thành phần tốt nhất
idx = np.argsort(-vals)
vals = vals[idx]
vecs = vecs[:, idx]

# 5. Lấy k=1 vector đầu tiên
k = 1
U_reduce = vecs[:, :k]
print(f"Vector thành phần chính (Ureduce):\n{U_reduce}")

# --- YÊU CẦU: Chiếu dữ liệu gốc lên thành phần chính, lưu vào Z ---
# Công thức: Z = X * U_reduce
Z = np.dot(X_center, U_reduce)

# --- YÊU CẦU: Phục hồi lại X dùng Z và Ureduce, lưu vào X_restore ---
# Công thức: X_restore = Z * U_reduce.T
# Lưu ý: Phải cộng lại 'mean' vì lúc đầu đã trừ đi
X_restore = np.dot(Z, U_reduce.T) + mean

# --- YÊU CẦU: Tính sự khác biệt trung bình (Sai số tái tạo) ---
# GỢI Ý: 1/m * sum(||x - x_restore||^2)
# GỢI Ý: Dùng np.sum() hoặc vòng lặp for
diff = X - X_restore
error = np.mean(np.sum(diff**2, axis=1))
print(f"Sự khác biệt trung bình (Reconstruction Error): {error}")

# --- YÊU CẦU: Tính tổng phương sai của dữ liệu ---
total_variance = np.mean(np.sum((X - mean)**2, axis=1))
print(f"Tổng phương sai (Total Variance): {total_variance}")

# --- KIỂM TRA TỈ LỆ (Phần đọc thêm) ---
print(f"Tỉ lệ sai số / tổng phương sai: {error/total_variance}")
print(f"Tỉ lệ lambda bị mất (lambda_2 / sum): {vals[1]/np.sum(vals)}")

# --- VẼ HÌNH MINH HỌA KẾT QUẢ ---
# Vẽ điểm phục hồi màu xanh hình sao ('b*')
plt.plot(X_restore[:, 0], X_restore[:, 1], 'b*', label='Dữ liệu phục hồi')

# Vẽ đường nối giữa điểm gốc và điểm phục hồi để dễ hình dung
for p1, p2 in zip(X, X_restore):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.1)

plt.title('Minh hoạ PCA: Chiếu dữ liệu xuống 1 chiều')
plt.axis('equal') # Quan trọng: Trục x, y tỉ lệ 1:1 mới thấy góc vuông
plt.show()