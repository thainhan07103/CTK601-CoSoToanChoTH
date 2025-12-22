import numpy as np  
import matplotlib.pyplot as plt

text_file = open("XuLyDuLieu/ex1data1.txt", "r")
lines = text_file.readlines()

data = np.array([[float(x) for x in line.split(",")] for line in lines])
text_file.close()

m, n = data.shape

X = data[:, range(n-1)]
Y = data[:, -1]

X = np.insert(X, 0, 1, axis=1)

#PP tìm chính xác
A = np.dot(X.T, X)
theta1 = np.dot(np.dot(np.linalg.inv(A), X.T), Y)

#PP Gradient Descent
theta2 = np.zeros(X.shape[1])
alpha = 0.01
nb_it = 2000
for i in range(nb_it):
    h = np.dot(X, theta2)
    error = h - Y
    gradient = (1/m)* np.dot(X.T, error)
    theta2 = theta2 - alpha * gradient
    
#So sánh kết quả của 2 theta
print("Theta from Normal Equation:", theta1)
print("Theta from Gradient Descent:", theta2)

#Vẽ dữ liệu và đường hồi quy lên cùng một đồ thị
y_pred = np.dot(X, theta2)

plt.plot(data[:, 0], data[:, -1], "ro")
plt.plot(data[:, 0], y_pred)

plt.xlabel("Area")
plt.ylabel("Price")
plt.show()