import math
import numpy as np
import matplotlib.pyplot as plt

def strategy(sigma_i,sigma_j,n):
    s = []
    v = []

    if (1 + math.pow(-1, n)) / 2 == 0:
        s.append([1, 0])
        v.append([math.pow(sigma_i, n - 1), 0])
    else:
        s.append([0, 1])
        v.append([0, math.pow(sigma_j, n - 1)])
    for i in range(1, n):
        if (1 + math.pow(-1, n - i + 1)) / 2 == 0:
            temp_s = [sigma_i * s[-1 - i + 1][0], 1 - sigma_i * s[-1 - i + 1][0]]
            s.insert(0, temp_s)
            temp_v = (np.array(temp_s) * np.array([sigma_i, sigma_j]) ** (n - 1 - i)).tolist()
            v.insert(0, temp_v)
        else:
            temp_s = [1 - sigma_j * s[-1 - i + 1][1], sigma_j * s[-1 - i + 1][1]]
            s.insert(0, temp_s)
            temp_v = (np.array(temp_s) * np.array([sigma_i, sigma_j]) ** (n - 1 - i)).tolist()
            v.insert(0, temp_v)
    return s,v

def pow_x_2_0_1(x):
    return math.pow(x,2)

n=15

sigma_i=0.6
sigma_j=0.8

sigma_ij = sigma_ji = 0.3

tol_i = 1-pow(sigma_i,2)
tol_j= 1-pow(sigma_j,2)

v = None
flag=n%2

alpha=None
beta=None

truth = strategy(sigma_i, sigma_j, n)[1][-1]
for turn in range(n):
    s_i, v_i = strategy(sigma_i, sigma_ij, n)
    s_j, v_j = strategy(sigma_ji, sigma_j, n)

    if turn % 2 == 1:
        if v_i[turn][1] + tol_j*v_j[turn][1] >= v_j[turn][1]:
            v = [v_i[turn][0], v_i[turn][1]]
            print(turn+1)
            break
        else:
            # 当总轮数为奇数时，i为最终决策者，在奇数轮时，若i的方案被拒绝，i提高对j的衰减系数预期，j会提高自己的容忍度以在i下次决策时不至于让自己走进拖延的窘境
            if flag==1:
                tol_j=tol_j*math.pow()+1-tol_j
                sigma_ij = sigma_ij*math.pow(turn/n+1,2)+1-sigma_ij

            # 当总轮数为偶数时，j为最终决策者，在奇数轮时，若i的方案被拒绝，i提高对j的衰减系数预期，j会降低自己的容忍度以在i下次决策时让自己尽可能多的获得利益
            else:
                tol_j=tol_j*(1-math.pow())
                sigma_ij = sigma_ij*math.pow(turn/n,2)+1-sigma_ij


    else:
        if v_j[turn][0] + tol_i*v_i[turn][0] >= v_i[turn][0]:
            v = [v_j[turn][0], v_j[turn][1]]
            print(turn+1)
            break
        else:
            # 当总轮数为奇数时，i为最终决策者，在偶数轮时，若j的方案被拒绝，j提高对i的衰减系数预期，i会降低自己的容忍度以在i下次决策时让自己尽可能多的获得利益
            if flag==1:
                tol_i=tol_i*(1-math.pow())
                sigma_ji = sigma_ji*math.pow(turn/n,2)+1-sigma_ji

            # 当总轮数为偶数时，j为最终决策者，在偶数轮时，若j的方案被拒绝，j提高对i的衰减系数预期，i会提高自己的容忍度以在j下次决策时不至于让自己走进拖延的窘境
            else:
                tol_i=tol_i*math.pow(turn/n,2)+1-tol_i
                sigma_ji = sigma_ji*math.pow(turn/(n+1),2)+1-sigma_ji

print(v)

'''
sigma_i=0.8
sigma_j=0.3

init=0.3'''


'''
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
'''