import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from chpt3.neural_network import softmax
from chpt4.learning import cross_entropy_error_with_one_hot

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()      # 复制一份输入，避免修改原始输入
        out[self.mask] = 0  # 将输入的x中小于0的位置置0，其他位置不变

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(x))   # 不使用self.out，防止后续计算直接修改该类中的out值
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)

        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error_with_one_hot(self.y, self.t)

        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size     # 保证梯度不随batch增大而增大

        return dx
    
class BatchNormalization:
    def __init__(self):
        pass

    def forward(self, x):
