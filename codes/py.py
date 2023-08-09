import numpy as np
import math
omat = np.array([[0,1],[-1,0]]) 
A = np.array([1,-1])
B = np.array([-4,6])
C = np.array([-3,-5])
print("A=",A,end=" ")
print("B=",B,end=" ")
print("C=",C)

#print("Since D is midpoint of BC")
D=(C+B)/2
print("D = (C + B)/2 =",end=" ")
print(D)

#print("Since E is midpoint of CA")
E=(C+A)/2
print("E = (C + A)/2 =",end=" ")
print(E)

#print("Since F is midpoint of AB")
F=(A+B)/2
print("F = (A + B)/2 =",end=" ")
print(F,end="\n")

def dir_vec(A,B):
 return B-A
#print("The direction vector AD:",end=" ")
#print(dir_vec(A,D))
#print("The direction vector BE:",end=" ")
#print(dir_vec(B,E))

def norm_vec(A,B):
  return omat@dir_vec(A,B)
  #return np.matmul(omat, dir_vec(A,B)
#print(norm_vec(A,D))
#print(norm_vec(B,E))  
n1=norm_vec(A,D)
n2=norm_vec(B,E)
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
#print(line_gen(A,D))
#print(line_gen(B,E))

def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  return P
G=line_intersect(n1,A,n2,B)
print("The centroid is:",end=" ")
print(G)

AG=math.sqrt((G[0]-A[0])**2+(G[1]-A[1])**2)
GD=math.sqrt((G[0]-D[0])**2+(G[1]-D[1])**2)

BG=math.sqrt((G[0]-B[0])**2+(G[1]-B[1])**2)
GE=math.sqrt((G[0]-E[0])**2+(G[1]-E[1])**2)

CG=math.sqrt((G[0]-C[0])**2+(G[1]-C[1])**2)
GF=math.sqrt((G[0]-F[0])**2+(G[1]-F[1])**2)

print("AG/GD= "+str(AG/GD))
print("BG/GE= "+str(BG/GE))
print("CG/GF= "+str(CG/GF))


