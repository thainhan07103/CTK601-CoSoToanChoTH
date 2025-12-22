import numpy as np

text_file = open("XuLyDuLieu/ex1data2.txt", "r")
lines = text_file.readlines()
data = np.array([[float(x) for x in line.split(",")] for line in lines])
text_file.close()

m, n = data.shape