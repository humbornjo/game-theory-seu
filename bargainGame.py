import math
import numpy as np
import matplotlib.pyplot as plt

n=15
sigma_i=0.8
sigma_j=0.8
s=[]
v=[]

if (1+math.pow(-1,n))/2==0:
    s.append([1,0])
    v.append([math.pow(sigma_i,n-1),0])
else:
    s.append([0,1])
    v.append([0,math.pow(sigma_j, n - 1)])
for i in range(1,n):
    if (1+math.pow(-1,n-i+1))/2==0:
        temp_s=[sigma_i*s[-1-i+1][0],1-sigma_i*s[-1-i+1][0]]
        s.insert(0,temp_s)
        temp_v=(np.array(temp_s)*np.array([sigma_i,sigma_j])**(n - 1-i)).tolist()
        v.insert(0,temp_v)
    else:
        temp_s=[1-sigma_j*s[-1-i+1][1],sigma_j*s[-1-i+1][1]]
        s.insert(0,temp_s)
        temp_v=(np.array(temp_s)*np.array([sigma_i,sigma_j])**(n - 1-i)).tolist()
        v.insert(0,temp_v)

x=range(1,n+1)

plt.subplot(2, 1, 1)
s_i=[s[i][0] for i in range(len(s))]
s_j=[s[i][1] for i in range(len(s))]
plt.plot(x,s_i)
plt.plot(x,s_j)
plt.xlabel('turns')
plt.ylabel('ratio')
plt.xlim(1,n)
plt.ylim(0,1)

plt.subplot(2, 1, 2)
v_i=[v[i][0] for i in range(len(v))]
v_j=[v[i][1] for i in range(len(v))]
plt.plot(x,v_i)
plt.plot(x,v_j)
plt.xlabel('turns')
plt.ylabel('value')
plt.xlim(1,n)
plt.ylim(0,1)
plt.show()