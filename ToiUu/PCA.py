import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Đọc tập dữ liệu “X.csv lưu vào biến X. Tập dữ liệu này chỉ có 2 cột (n=2) [cite: 207] ---
df = pd.read_csv('XuLyDuLieu/X.csv', header=None)
X = df.values
m, n = X.shape

# --- 2. Vẽ tập dữ liệu X dùng X[:,0] là trục ngang, X[:,1] là trục dọc [cite: 208] ---
# (Lưu ý: Để vẽ chồng lên nhau ở bước cuối, ta thường vẽ chung một lúc, 
# nhưng theo đề bài thì bước này chỉ yêu cầu vẽ X).
plt.plot(X[:, 0], X[:, 1], 'ro')

# --- 3. Tìm 1 thành phần chính của X (k=1), lưu vào Ureduce [cite: 210] ---
# Bước 3.1: Quy tâm dữ liệu (đưa tâm của X về gốc tọa độ) [cite: 184]
mean = np.mean(X, axis=0)
X_centered = X - mean

# Bước 3.2: Tính ma trận phương sai – hiệp phương sai C = (1/m) * X^T * X [cite: 186, 187]
# Yêu cầu dùng np.dot cho phép nhân ma trận
C = (1/m) * np.dot(X_centered.T, X_centered)

# Bước 3.3: Tìm giá trị riêng và vector riêng [cite: 188]
eigenvalues, eigenvectors = np.linalg.eig(C)

# Bước 3.4: Sắp xếp các giá trị riêng theo thứ tự giảm dần [cite: 190]
index = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[index]
eigenvectors = eigenvectors[:, index]

# Bước 3.5: Giữ lại k cột đầu tiên của U (k=1) lưu vào Ureduce [cite: 194, 195]
k = 1
Ureduce = eigenvectors[:, 0:k]
print("Ureduce:\n", Ureduce)

# --- 4. Chiếu dữ liệu gốc lên thành phần chính này, lưu kết quả vào Z [cite: 211] ---
# Công thức: Z = X * Ureduce [cite: 200]
Z = np.dot(X_centered, Ureduce)
print("\nZ (5 giá trị đầu):\n", Z[:5])

# --- 5. Phục hồi lại X bằng cách sử dụng Z và Ureduce, lưu kết quả vào Xrestore [cite: 212] ---
# Công thức: X_restore = Z * Ureduce^T [cite: 203]
# Lưu ý: Kết quả phép nhân là dữ liệu đã quy tâm, cần cộng lại mean ban đầu.
X_restore_centered = np.dot(Z, Ureduce.T)
Xrestore = X_restore_centered + mean
print("\nXrestore (5 giá trị đầu):\n", Xrestore[:5])

# --- 6. Tính sự khác biệt trung bình giữa X và Xrestore [cite: 213] ---
# Công thức: (1/m) * sum(||x - x_restore||^2) [cite: 215]
diff = X - Xrestore
diff_squared = diff ** 2
# Dùng np.sum để tính tổng các phần tử rồi chia cho m
mean_error = np.sum(diff_squared) / m
print("\nSự khác biệt trung bình:", mean_error)

# --- 7. Vẽ dữ liệu gốc X và dữ liệu phục hồi Xrestore lên cùng đồ thị [cite: 222] ---
# Vẽ dữ liệu gốc X (màu đỏ, chấm tròn)
plt.plot(X[:, 0], X[:, 1], 'ro')

# Vẽ dữ liệu phục hồi Xrestore (màu xanh, dấu cộng)
plt.plot(Xrestore[:, 0], Xrestore[:, 1], 'b+')

# (Tùy chọn) Vẽ đường nối giữa điểm gốc và điểm phục hồi để thấy rõ hình chiếu
for i in range(m):
    plt.plot([X[i, 0], Xrestore[i, 0]], [X[i, 1], Xrestore[i, 1]], 'k--', linewidth=0.5)

plt.axis('equal') # Đảm bảo tỉ lệ trục để thấy góc vuông
plt.show()