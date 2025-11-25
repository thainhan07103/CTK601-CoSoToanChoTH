#Bài tập cực trị của hàm 1 biến fx = ae^-0.2x
from math import *
def f(x):
    return x*exp(-0.2*x)

a = -100
b = 100

r = (5.0**0.5 - 1.0)/2
for i in range(100):
    c = a + r*(b - a)
    d = b + r*(a - b)
    if f(c) > f(d):
        a = d
    else:
        b = c

x = (a + b)/2
print("Điểm cực trị của hàm số là x =", x)
