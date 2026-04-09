import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)    # 支持数组形式输入

def sigmoid(x):             # 隐藏网络神经元的激活函数
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def identity_function(x):   # 回归问题的激活函数
    return x

def softmax(a):             # 分类问题的激活函数
                            # softmax的输出数组的总和恒为1 —— 输出可解释为“概率”
    c = np.max(a)
    exp_a = np.exp(a - c)   # 防止数据溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

if __name__ == "__main__":
    x = np.arange(-5, 5, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(bottom=-0.1, top=1.1) # 指定y轴范围
    # plt.show()

    x = np.arange(-5, 5, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1,1)
    # plt.show()

    X = np.array([1, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5],
                   [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    # print(Z1)

    W2 = np.array([[0.1, 0.4],
                    [0.2, 0.5],
                   [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    # print(Z2)

    W3 = np.array([[0.1, 0.3],
                   [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)

    # Conclusion
    def init_network():
        network = {}
        network['W1'] = np.array([[0.1, 0.3, 0.5],
                                 [0.2, 0.4, 0.6]])
        network['b1'] = np.array([0.1, 0.2, 0.3])
        network['W2'] = np.array([[0.1, 0.4],
                                   [0.2, 0.5],
                                  [0.3, 0.6]])
        network['b2'] = np.array([0.1, 0.2])
        network['W3'] = np.array([[0.1, 0.3],
                                   [0.2, 0.4]])
        network['b3'] = np.array([0.1, 0.2])

        return network

    def forward(network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = identity_function(a3)

        return y

    network = init_network()
    x = np.array([1, 0.5])
    y = forward(network, x)
    print(y)