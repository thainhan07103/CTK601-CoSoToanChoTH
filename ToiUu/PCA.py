import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# BÀI TẬP 4: PHÂN TÍCH THÀNH PHẦN CHÍNH (PCA) 
# ============================================================

# ------------------------------------------------------------
# Bước 1: Đọc tập dữ liệu "X.csv" lưu vào biến X [cite: 1370]
# ------------------------------------------------------------
# Dùng pandas đọc file csv không có header
df = pd.read_csv('XuLyDuLieu/X.csv', header=None)
# Chuyển về dạng numpy array để tính toán
X = df.values 
m, n = X.shape
print(f"Đã đọc dữ liệu: {m} dòng, {n} cột.")

# ------------------------------------------------------------
# Bước 2: Vẽ tập dữ liệu X [cite: 1371]
# ------------------------------------------------------------
# Dùng X[:,0] là trục ngang, X[:,1] là trục dọc
plt.figure(figsize=(8, 8))
plt.plot(X[:, 0], X[:, 1], 'ro', label='Dữ liệu gốc') # 'ro' là chấm đỏ [cite: 1372]
plt.title('Bài tập 4: PCA')

# ------------------------------------------------------------
# Bước 3: Tìm 1 thành phần chính của X (k=1) [cite: 1373]
# ------------------------------------------------------------
# Để làm bước này, ta cần thực hiện các bước nhỏ trong phần lý thuyết[cite: 1345]:

# 3a. Quy tâm dữ liệu: đưa tâm của X về gốc toạ độ [cite: 1347]
mean = np.mean(X, axis=0)
X_centered = X - mean # Biến X đã quy tâm

# 3b. Tính ma trận phương sai - hiệp phương sai C = (1/m) * X^T * X [cite: 1349, 1350]
C = (1/m) * (X_centered.T @ X_centered)

# 3c. Tìm giá trị riêng L và vector riêng U của C [cite: 1351]
L, U = np.linalg.eig(C)

# 3d. Sắp xếp các giá trị riêng theo thứ tự giảm dần [cite: 1353]
# np.argsort trả về chỉ số, thêm dấu trừ để sort giảm dần
index = np.argsort(-L)
L = L[index]
U = U[:, index]

print("Các giá trị riêng (L):", L)
print("Các vector riêng (U):", U)

# 3e. Giữ lại k cột đầu tiên của U (k=1) lưu vào Ureduce [cite: 1357, 1373]
k = 1
Ureduce = U[:, 0:k]
print(f"Ma trận Ureduce (k={k}):\n", Ureduce)

# ------------------------------------------------------------
# Bước 4: Chiếu dữ liệu gốc lên thành phần chính, lưu vào Z [cite: 1374]
# ------------------------------------------------------------
# Công thức: Z = X * Ureduce [cite: 1363]
Z = X_centered @ Ureduce

# ------------------------------------------------------------
# Bước 5: Phục hồi lại X bằng cách sử dụng Z và Ureduce, lưu vào Xrestore [cite: 1375]
# ------------------------------------------------------------
# Công thức: X_restore = Z * Ureduce^T [cite: 1366]
# Lưu ý: Kết quả này là dữ liệu đã quy tâm, muốn về dữ liệu gốc ban đầu cần cộng lại mean
X_restore_centered = Z @ Ureduce.T
X_restore = X_restore_centered + mean 

# ------------------------------------------------------------
# Bước 6: Tính sự khác biệt trung bình giữa X và Xrestore [cite: 1376]
# ------------------------------------------------------------
# Công thức: (1/m) * sum(||x - x_restore||^2)
diff = X - X_restore
# Dùng np.sum tính tổng bình phương cho từng dòng, sau đó mean 
error = np.mean(np.sum(diff**2, axis=1)) 
print(f"Sự khác biệt trung bình (Reconstruction Error): {error}")

# ------------------------------------------------------------
# Bước 7: Tính tổng phương sai của dữ liệu (gốc) [cite: 1380]
# ------------------------------------------------------------
# Công thức: (1/m) * sum(||x||^2) - Áp dụng trên dữ liệu đã quy tâm
total_variance = np.mean(np.sum(X_centered**2, axis=1))
print(f"Tổng phương sai (Total Variance): {total_variance}")

# ------------------------------------------------------------
# Bước 8: Tính tỉ lệ và so sánh [cite: 1381, 1382]
# ------------------------------------------------------------
# Tính tỉ lệ lỗi / tổng phương sai
ratio_error = error / total_variance
print(f"Tỉ lệ (Lỗi / Tổng phương sai): {ratio_error}")

# So sánh với tỉ lệ lambda bị mất (lambda_2 / tổng lambda) [cite: 1384]
# Vì giữ lại k=1 (lambda_1), phần mất đi là lambda_2
ratio_lambda_loss = L[1] / np.sum(L)
print(f"Tỉ lệ Lambda bị mất (L[1] / Sum(L)): {ratio_lambda_loss}")
print("=> Nhận xét: Hai tỉ lệ này xấp xỉ bằng nhau.")

# ------------------------------------------------------------
# Bước 9: Vẽ dữ liệu gốc X và dữ liệu phục hồi Xrestore lên cùng đồ thị [cite: 1385]
# ------------------------------------------------------------
plt.plot(X_restore[:, 0], X_restore[:, 1], 'b*', label='Dữ liệu phục hồi') # Vẽ Xrestore
plt.legend()
plt.axis('equal') # Chỉnh tỉ lệ trục cho vuông góc
plt.grid(True)
plt.show()