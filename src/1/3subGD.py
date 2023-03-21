import numpy as np
import matplotlib.pyplot as plt
import math

# data
p = 10
alpha = 0.001
epsilon = 1e-5  # 最大允许误差

A = {}
x = np.zeros((300,1))
e = {}
b = {}

# 获取数据
def get_data():
    for i in range(20):
        name_A = "./data/A/A" + str(i) + ".npy"
        name_e = "./data/e/e" + str(i) + ".npy"
        name_b = "./data/b/b" + str(i) + ".npy"
        A[i] = np.load(name_A)
        e[i] = np.load(name_e)
        b[i] = np.load(name_b)

def A_node(xk, i):
    gradient_i = A[i].T.dot(A[i].dot(xk)-b[i])
    return gradient_i

def master():
    get_data()
    x = np.load("./data/x.npy")
    xk = np.zeros((300,1))
    k = 0
    a = alpha
    xk_step = []    # 纪录每一步的结果
    while 1:
        k = k + 1
        a = a / math.sqrt(k)
        partial_gradient = 0    # 光滑部分梯度
        for i in range(20):
            partial_gradient = partial_gradient + A_node(xk, i)
        L1_gradient = np.zeros((300,1))
        for i in range(300):
            if xk[i] > 0:
                L1_gradient[i] = 1
            elif xk[i] < 0:
                L1_gradient[i] = -1
            else:
                L1_gradient[i] = np.random.uniform(-1,1)

        gradient = partial_gradient + L1_gradient
        xk_1 = xk - a * gradient

        xk_step.append(xk_1)

        if np.linalg.norm(xk_1-xk) < epsilon:
        # if k > 6000:
            break
        else:
            if k%500 == 0:
                print(k, np.linalg.norm(xk_1 - xk))
            xk = xk_1.copy()
    
    x_optm = xk_1.copy()    # 跳出循环得到最优解
    return xk_step, x_optm

def draw(xk_step, x_optm):
    x = np.load("./data/x.npy")
    dis_xk_real = []    # xk与真实值的距离
    dis_xk_optm = []    # xk与最优解的距离

    for xk in xk_step:
        dis_xk_real.append(np.linalg.norm(xk - x))        # 记录每步计算结果与真值的距离
        dis_xk_optm.append(np.linalg.norm(xk - x_optm))   # 记录每步计算结果与最优解的距离

    # 绘图
    plt.title('subgradient method')
    plt.xlabel('k:Iteration_times')
    plt.ylabel('distance')

    plt.plot(dis_xk_real, 'r', label='distance from truth value to xk')
    plt.plot(dis_xk_optm, 'b', label='distance from the optimal solution to xk')

    plt.grid()
    plt.legend()
    plt.show()


xk_step, x_optm = master() 
draw(xk_step, x_optm)