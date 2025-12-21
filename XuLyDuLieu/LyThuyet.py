import numpy as np
import matplotlib.pyplot as plt

text_file = open("XuLyDuLieu/ex1data1.txt", "r")
lines = text_file.readlines()

#Đọc dữ liệu từ file và chuyển đổi thành mảng numpy
data = np.array([[float(x) for x in line.split(',')] for line in lines])
m, n = data.shape
print(data)

# Vẽ biểu đồ dữ liệu
plt.plot(data[:, 0], data[:, 1], "ro")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# Tính toán tham số theta bằng phương pháp chính xác
X = data[:, range(n - 1)]
y = data[:, -1]
X = np.insert(X, 0, 1, axis=1)  # Thêm cột bias
A = np.dot(X.T, X)
theta = np.dot(np.dot(np.linalg.inv(A), X.T), y)
theta, _, _, _ = np.linalg.lstsq(X, y)
print("Theta found by normal equation:", theta)

#Tính bằng phương pháp gradient descent
theta = np.zeros(n)
alpha = 0.01
nb_it = 1500

for it in range(nb_it):
    # gradient = np.dot(X.T, (np.dot(X, theta) - y)) / m
    gradient =(np.dot(np.dot(X.T, X), theta) - np.dot(X.T, y)) / m
    theta = theta - alpha * gradient
print("Theta found by gradient descent:", theta)