import numpy as np
import matplotlib.pyplot as plt
import math

# data
p = 0.01
c = 0.005
epsilon = 7*1e-4  # 最大允许误差

A = {}
x = np.zeros((300,1))
e = {}
b = {}

ATb = []
ATA_add_c_inv = [] # 这两个值在进行计算的时候无论迭代多少次都不会变

# 获取数据
def get_data():
    for i in range(20):
        name_A = "./data/A/A" + str(i) + ".npy"
        name_e = "./data/e/e" + str(i) + ".npy"
        name_b = "./data/b/b" + str(i) + ".npy"
        A[i] = np.load(name_A)
        e[i] = np.load(name_e)
        b[i] = np.load(name_b)

def A_node_ATmul(i, c):
    if c=='A':
        result = A[i].T.dot(A[i])
    elif c=='b':
        result = A[i].T.dot(b[i])
    return result

def A_node_xk_1(yk, lk_i, i):
    xk_1_i = ATA_add_c_inv[i].dot(ATb[i] + c*yk - lk_i)
    return xk_1_i

def A_node_lk_1(lk_i, xk_1_i, yk_1):
    lk_1_i = lk_i + c*(xk_1_i - yk_1)
    return lk_1_i

def master():
    get_data()
    xk = []
    yk = np.zeros((300,1))
    lk = []
    for i in range(20):
        xk.append(np.zeros((300,1)))
        lk.append(np.zeros((300,1)))
    yk_1 = np.zeros((300,1))
    xk_step_0 = []
    xk_1 = []
    lk_1 = []

    ATA = []
    count = 0
    
    for i in range(20):# 这个值是固定的，在开始时计算一次，不需要重复计算
        ATA.append(A_node_ATmul(i,'A'))
        ATb.append(A_node_ATmul(i,'b'))
    I = np.eye(300,300)
    for i in range(20):
        ATA_add_c_inv.append( np.linalg.inv(ATA[i]+c*I) )
    
    while 1:    # 开始迭代
        count = count + 1
        for i in range(20):
            xk_1.append( A_node_xk_1(yk, lk[i], i) )

        # 计算y的更新值
        for i in range(300):
            temp = 0
            for j in range(20):
                temp = temp + (xk_1[j][i]+lk[j][i]/c)     # 这个值要多次用到，防止重复计算

            if temp < -p/c:
                yk_1[i] = (temp + p/c)/20
            elif temp > p/c:
                yk_1[i] = (temp - p/c)/20
            else:
                yk_1[i] = 0
        
        # 计算\lambda的更新值
        for i in range(20): # 将值传入每个节点并将结果返回
            lk_1.append( A_node_lk_1(lk[i], xk_1[i], yk_1) )

        xk_step_0.append(xk_1[0])

        if np.linalg.norm(xk_1[0]-xk[0]) < epsilon and np.linalg.norm(lk_1[0]-lk[0])<epsilon and np.linalg.norm(yk_1-yk) < epsilon: # 因为在计算中无法准确找到次梯度为0的点，我们近似地两步之间的二范数10^{-5},看作结果收敛
            print(np.linalg.norm(xk_1[0]-xk[0]))
            print(count, np.linalg.norm(xk_1[0] - xk[0]), np.linalg.norm(lk_1[0] - lk[0]), np.linalg.norm(yk_1 - yk))
            break
        else:
            # 更新值
            if count%500 == 0:
                print(count, np.linalg.norm(xk_1[0] - xk[0]), np.linalg.norm(lk_1[0] - lk[0]), np.linalg.norm(yk_1 - yk))
            xk = xk_1.copy()
            yk = yk_1.copy()
            lk = lk_1.copy()
            xk_1 = []
            lk_1 = []
        
        
    x_optm = xk_1[0].copy()  # 跳出循环得到最优解
    return xk_step_0, x_optm

def draw(xk_step, x_optm):
    x = np.load("./data/x.npy")
    dis_xk_real = []    # xk与真实值的距离
    dis_xk_optm = []    # xk与最优解的距离

    for xk in xk_step:
        dis_xk_real.append(np.linalg.norm(xk - x))        # 记录每步计算结果与真值的距离
        dis_xk_optm.append(np.linalg.norm(xk - x_optm))   # 记录每步计算结果与最优解的距离

    # 绘图
    plt.title('Alternating Direction Multiplier Method(ADMM)')
    plt.xlabel('k:Iteration_times')
    plt.ylabel('distance')

    plt.plot(dis_xk_real, 'r', label='distance from truth value to xk')
    plt.plot(dis_xk_optm, 'b', label='distance from the optimal solution to xk')

    plt.grid()
    plt.legend()
    plt.show()


xk_step, x_optm = master() 
draw(xk_step, x_optm)
    