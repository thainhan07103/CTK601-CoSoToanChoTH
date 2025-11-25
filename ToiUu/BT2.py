#Tìm x sao cho hàm bên dưới đạt giá lớn nhất trong khoảng [0, 1]
# x^5 - 10x^2 + 2x
#Sử dụng Phương pháp II (đạo hàm và tìm kiếm nhị phân) và III (Tìm kiếm theo tỉ lệ vàng)
def fDH(x):
    return 5*x**4 - 20*x + 2
def f(x):
    return x**5 - 10*x**2 + 2*x

#Phương pháp 2: Tìm kiếm nhị phân
a1 = 0
b1 = 1

for i in range(100):
    m = (a1+b1)/2
    if fDH(m) * fDH(b1) > 0:
        b1 = m
    else:
        a1 = m

x1 = (a1 + b1)/2
print("Phương pháp 1: x =", x1)

#Phương pháp 3: Tìm kiếm theo tỉ lệ vàng
a2 = 0
b2 = 1
r = (5.0**0.5 - 1.0)/2
for i in range(100):
    c = a2 + r*(b2-a2)
    d = b2 + r*(a2-b2)
    if f(c) > f(d):
        a2 = d
    else:
        b2 = c

x2 = (a2 + b2)/2
print("Phương pháp 2: x =", x2)

