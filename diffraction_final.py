import numpy as np
import pandas as pd
import numpy.linalg as lin
from numpy import sqrt,cos,sin,pi
import matplotlib.pyplot as plt
import sys
phi,theta=sys.argv[1:]
phi=int(phi)/180*pi
theta=int(theta)/180*pi
a=5.431
lmin=0.6
lmax=1.2
def neg(a):
    b=-1*np.copy(a)
    b[-1]=-a[-2]
    b[-2]=-a[-1]
    return b
def ero2d(a):
    a=np.copy(a).astype("float")
    if a[0,0]*a[1,0]>0:
        a[1]=a[1]+neg(a[0])/a[0,0]*a[1,0]
    else:
        a[1]=a[1]-a[0]/a[0,0]*a[1,0]
    if a[1,1]>0:
        a[1]=a[1]/a[1,1]
    else:
        a[1]=-neg(a[1])/a[1,1]
    if a[0,1]>0:
        a[0]=a[0]+neg(a[1])*a[0,1]
    else:
        a[0]=a[0]-a[1]*a[0,1]
    if a[0,0]>0:
        a[0]=a[0]/a[0,0]
    else:
        a[0]=-neg(a[0])/a[0,0]
    return a
def ero3d(a):
    a=np.copy(a).astype("float")
    if a[0,0]*a[1,0]>0:
        a[1]=a[1]+neg(a[0])/a[0,0]*a[1,0]
    else:
        a[1]=a[1]-a[0]/a[0,0]*a[1,0]
    
    if a[0,0]*a[2,0]>0:
        a[2]=a[2]+neg(a[0])/a[0,0]*a[2,0]
    else:
        a[2]=a[2]-a[0]/a[0,0]*a[2,0]
    a[1:,1:]=ero2d(a[1:,1:])
    if a[0,1]>0:
        a[0]=a[0]+neg(a[1])*a[0,1]
    else:
        a[0]=a[0]-a[1]*a[0,1]
    
    if a[0,2]>0:
        a[0]=a[0]+neg(a[2])*a[0,2]
    else:
        a[0]=a[0]-a[2]*a[0,2]
    
    if a[0,0]>0:
        a[0]=a[0]/a[0,0]
    else:
        a[0]=-neg(a[0])/a[0,0]
    return a[:,3:]    


def selection_rule(h,k,l):
    if not (h or k or l):
        return 0
    elif h%2 and k%2 and l%2:
        return 1
    elif not (h%2 or k%2 or l%2):
        if (h+k+l)%4==0:
            return 1
        else:
            return 0
    else:
        return 0
C=np.array([[1/sqrt(3),1/sqrt(2),1/sqrt(6)],
   [1/sqrt(3),-1/sqrt(2),1/sqrt(6)],
   [1/sqrt(3),0,-2/sqrt(6)]]) 

M1=np.array([[cos(theta),0,-sin(theta)],
             [0,1,0],
             [sin(theta),0,cos(theta)]])
M2=np.array([[cos(phi),-sin(phi),0],
            [sin(phi),cos(phi),0],
            [0,0,1]])

mat=np.matmul(np.matmul(M1,M2),lin.inv(C))/a
print(mat)
min_max=np.array([[0,-1,-1],[2,1,1]]).T/lmin
constraint=np.concatenate([mat,min_max],axis=1)
con=ero3d(constraint)
reciprocal=[]
for i in range(int(con[0,0]),int(con[0,1])+1):
    for j in range(int(con[1,0]),int(con[1,1])+1):
        for k in range(int(con[2,0]),int(con[2,1])+1):
            if selection_rule(i,j,k):
                reciprocal.append([i,j,k])
reciprocal=np.array(reciprocal)
C3=np.matmul(np.matmul(M1,M2),np.matmul(lin.inv(C),reciprocal.T)).T/a
l=2*C3[:,0]/np.sum(C3**2,axis=1)
x=[]
y=[]
ind=[]
d=9
for i,ele in enumerate(l):
    if ele>=lmin and ele<=lmax:
        k1=C3[i]-np.array([1/ele,0,0])
        if k1[0]>0:
            project=k1/k1[0]*d
            if abs(project)[1]<=5 and abs(project)[2]<=5:
                x.append(-project[1])
                y.append(project[2])
                ind.append(reciprocal[i])
                
plt.figure(figsize=(5,5))
plt.xlabel("cm")
plt.ylabel("cm")
plt.title("φ= %s⁰; θ= %s⁰"%(sys.argv[1],sys.argv[2]))
plt.scatter(x,y)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.savefig("phi_%s_theta_%s.png"%(sys.argv[1],sys.argv[2]),dpi=600)
#plt.show()





