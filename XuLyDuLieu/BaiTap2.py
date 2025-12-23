import numpy as np

text_file = open("XuLyDuLieu/ex1data2.txt", "r")
lines = text_file.readlines()
data = np.array([[float(x) for x in line.split(",")] for line in lines])
text_file.close()

m, n = data.shape

X = data[:, range(n-1)]
Y = data[:, -1]
X = np.insert(X, 0, 1, axis=1)

#Phương pháp tìm chính xác
A = np.dot(X.T, X)
theta = np.dot(np.dot(np.linalg.inv(A), X.T), Y)
print("Theta tìm được bằng phương pháp chính xác:", theta)

#Phương pháp Gradient Descent
theta = np.zeros(X.shape[1])
alpha = 0.01
nb_it = 2000

for i in range(nb_it):
    h = np.dot(X, theta)
    eror = h - Y
    gradient = (1/m)*np.dot(X.T, eror)
    theta = theta - alpha*gradient
print("Theta tìm được bằng phương pháp Gradient Descent:", theta)