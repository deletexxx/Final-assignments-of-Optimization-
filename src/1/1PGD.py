# b_i为 10 维的测量值
# A_i 为 10ⅹ300 维的测量矩阵
# x 为 300 维的未知稀疏向量且稀疏度为 5
# ei 为 10 维的测量噪声

# 设 x 的真值中的非零元素服从均值为 0 方差为 1 的高斯分布
# A_i 中的元素服从均值为 0 方差为 1 的高斯分布
# e_i 中的元素服从均值为 0 方差为 0.2 的高斯分布

import numpy as np
import matplotlib.pyplot as plt
import math

#data
alpha = 0.001
p = 100        # 正则化参数 10的时候最好然后11 然后1
epsilon = 1e-5  # 最大允许误差

A = {}
x = np.zeros((300,1))
e = {}
b = {}

# 生成数据
def Generate_data():
    # 稀疏矩阵x
    data_x = np.random.normal(0, 1, 5)
    col = np.random.randint(0, 300, 5)
    for i in range(5):
        x[col[i]] = data_x[i]
    np.save("./data/x", x)
    for i in range(20):
        A[i] = np.random.normal(0, 1, (10,300))
        e[i] = np.random.normal(0, math.sqrt(0.2), (10,1))
        b[i] = np.dot(A[i], x)+e[0]
        name_A = "./data/A/A" + str(i)
        name_e = "./data/e/e" + str(i)
        name_b = "./data/b/b" + str(i)
        np.save(name_A, A[i])
        np.save(name_e, e[i])
        np.save(name_b, b[i])

# 获取数据
def get_data():
    for i in range(20):
        name_A = "./data/A/A" + str(i) + ".npy"
        name_e = "./data/e/e" + str(i) + ".npy"
        name_b = "./data/b/b" + str(i) + ".npy"
        A[i] = np.load(name_A)
        e[i] = np.load(name_e)
        b[i] = np.load(name_b)
    x = np.load("./data/x.npy")

def A_node(xk,i):
    xk_half_temp = alpha*(A[i].T.dot(A[i].dot(xk)-b[i]))
    return xk_half_temp

def master():
    # Generate_data()
    get_data()
    xk_half_temp_sum = np.zeros((300,1))   # x^{k+1/2}_temp
    xk_half = np.zeros((300,1))   # x^{k+1/2}
    xk = np.zeros((300,1))            # xk
    xk_new = np.zeros((300,1))        # 每一步更新后的xk
    k = 0 # 记录迭代次数
    xk_step = []    # 记录每一步的计算结果
    while 1:
        xk_half_temp_sum = np.zeros((300,1))  # 重新更新时，要记得将和清零，否则会不断叠加
        for i in range(0,20):# 从master将xk分发到20个节点上计算x^{k+1/2}, 计数是0，……，19，仍是20个节点
            xk_half_temp_sum = xk_half_temp_sum + A_node(xk,i)# 将计算得到的部分x^{k+1/2}传回master，更新x^k,先进行求和

        xk_half = xk - xk_half_temp_sum
        for i in range(300):
            if xk_half[i] < -alpha*p:
                xk_new[i] = xk_half[i]+alpha*p
            elif xk_half[i] > alpha*p:
                xk_new[i] = xk_half[i]-alpha*p
            else:
                xk_new[i] = 0

        xk_step.append(xk_new.copy())  # 记录每一步的迭代结果,这里一定要进行深拷贝

        if np.linalg.norm(xk_new-xk) < epsilon: # 因为在计算中无法准确找到次梯度为0的点，我们近似地两步之间的二范数10^{-5},看作结果收敛
            break
        else:
            xk = xk_new.copy()  # 进行更新xk，注意一定要进行深拷贝，否则当xk_new改变时，xk也会改变
            k = k+1

    x_optm = xk_new.copy()  # 跳出循环得到最优解
    return xk_step, x_optm

def draw(xk_step, x_optm):
    x = np.load("./data/x.npy")
    dis_xk_real = []    # xk与真实值的距离
    dis_xk_optm = []    # xk与最优解的距离

    for xk in xk_step:
        dis_xk_real.append(np.linalg.norm(xk - x))        # 记录每步计算结果与真值的距离
        dis_xk_optm.append(np.linalg.norm(xk - x_optm))   # 记录每步计算结果与最优解的距离

    # 绘图
    plt.title('Proximal Gradient Method')
    plt.xlabel('k:Iteration_times')
    plt.ylabel('distance')

    plt.plot(dis_xk_real, 'r', label='distance from truth value to xk')
    plt.plot(dis_xk_optm, 'b', label='distance from the optimal solution to xk')

    plt.grid()
    plt.legend()
    plt.show()


xk_step, x_optm = master() 
draw(xk_step, x_optm)