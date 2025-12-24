import numpy as np
import pandas as pd

text_file = open("XuLyDuLieu/x.csv", "r")
lines = text_file.readlines()
data = np.array([[float(x) for x in line.split(",")] for line in lines])
text_file.close()