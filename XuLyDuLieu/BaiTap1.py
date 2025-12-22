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

A = np.dot(X.T, X)

