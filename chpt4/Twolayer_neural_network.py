import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from chpt1.differentiation import numerical_gradient
from chpt3.neural_network import sigmoid
from chpt4.learning import cross_entropy_error_with_one_hot

class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = sigmoid(a2)

        return y
    
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error_with_one_hot(y, t)
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)  # 返回函数loss_W = cross_entropy_error_with_one_hot(y, t)  # noqa: E731

        grads = {}
        grads = numerical_gradient(loss_W, self.params['W1'])
        grads = numerical_gradient(loss_W, self.params['b1'])
        grads = numerical_gradient(loss_W, self.params['W2'])
        grads = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy