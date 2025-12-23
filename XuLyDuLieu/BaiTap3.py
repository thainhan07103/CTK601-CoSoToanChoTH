import numpy as np

text_file = open("XuLyDuLieu/ex1data2.txt", "r")
lines = text_file.readlines()
data = np.array([[float(x) for x in line.split(",")] for line in lines])
text_file.close()

X = data[:, :-1]
y = data[:, -1]
m = len(y)

print(f"Dữ liệu gốc: {m} mẫu, {X.shape[1]} đặc trưng")
print(f"5 mẫu đầu tiên chưa chuẩn hóa: \n{X[:5]}")

# Chuẩn hóa dữ liệu
mean = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mean) / sigma

print("Đã chuẩn hóa dữ liệu")
print(f"Mean: {mean}")
print(f"Sigma: {sigma}")
print(f"5 mẫu đầu tiên sau khi chuẩn hóa: \n{X_norm[:5]}")

# Thêm cột 1
X_norm = np.insert(X_norm, 0, 1, axis=1)

#Phương pháp tìm chính xác với dữ liệu đã chuẩn hóa
A = np.dot(X_norm.T, X_norm)
theta1 = np.dot(np.dot(np.linalg.inv(A), X_norm.T), y)
print("Theta tìm được bằng phương pháp chính xác (chuẩn hóa):", theta1)

#Phương pháp Gradient Descent với dữ liệu đã chuẩn hóa
theta2 = np.zeros(X_norm.shape[1])
alpha = 0.01
nb_it = 2000
for i in range(nb_it):
    h = np.dot(X_norm, theta2)
    error = h - y
    gradient = (1/m)*np.dot(X_norm.T, error)
    theta2 = theta2 - alpha*gradient
    
print("Theta tìm được bằng phương pháp Gradient Descent (chuẩn hóa):", theta2)

#Tính chênh lệch
diff = np.abs(theta1 - theta2)
print("Chênh lệch giữa hai phương pháp (chuẩn hóa):", diff)
if np.all(diff < 0.1):
    print("\n=> KẾT LUẬN: Hai phương pháp đã cho kết quả tương đương nhau!")
else:
    print("\n=> KẾT LUẬN: Vẫn còn sai lệch, cần tăng số vòng lặp hoặc chỉnh alpha.")
    
# --- DỰ BÁO THỬ (Tùy chọn) ---
# Ví dụ: Dự báo giá nhà cho căn 1650 sq-ft, 3 phòng ngủ
# Ta phải chuẩn hóa input này trước khi nhân với theta
x_test = np.array([1650, 3])
x_test_norm = (x_test - mean) / sigma
# Thêm bias
x_test_norm = np.insert(x_test_norm, 0, 1) 

price_pred = np.dot(x_test_norm, theta2)
print(f"\nDự báo giá nhà (1650 sq-ft, 3 br): ${price_pred:,.2f}")
