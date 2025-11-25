#Bai tap buoi 3 chieu CN 23/11/2025
import numpy as  np
A = np.array([
    [2.0, 1],
    [1, 4]   
])

M = np.dot(A.T, A)
L, U = np.linalg.eig(M)
u = U[:,1]
z = np.dot(A, u)
squared_norm_z = np.dot(z, z)

print(squared_norm_z) 

M1 = np.dot(np.dot(U.T, np.diag(L)), U) #Kiem tra M1 voi M
print(M, M1)