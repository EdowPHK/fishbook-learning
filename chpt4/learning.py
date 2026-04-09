import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
from chpt1.differentiation import numerical_gradient
from chpt3.neural_network import softmax

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error_with_one_hot(y ,t, delta=1e-7):
    # log中增加微小数，防止y==0时计算无法进行
    if y.ndim == 1:
        t = t.reshape(1, t.size)    # .size会返回元素总数（行*列）
        y = y.reshape(1, y.size)    # 改为二维形式，便于统一用batch处理

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

def cross_entropy_error_without_one_hot(y, t, delta=1e-7):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size    # 高级索引

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error_with_one_hot(y, t)
        return loss

if __name__ == "__main__":
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    # print(mean_squared_error(np.array(y), np.array(t)))
    # print(cross_entropy_error(np.array(y), np.array(t)))
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
    
    # mini-batch
    train_size = x_train.shape[0]   # train_size == 60000
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    def function_2(x):
        return x[0]**2 + x[1]**2
    
    init_x = np.array([-3.0, 4.0])
    num = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
    # print(num)

    net = simpleNet()
    # print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    # print(p, np.argmax(p))
    t = np.array([0, 0, 1])
    # print(net.loss(x, t))

    def f(W):
        return net.loss(x, t)
    
    dW = numerical_gradient(f, net.W)
    print(dW)