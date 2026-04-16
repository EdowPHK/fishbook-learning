import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from collections import OrderedDict
import chpt5.layer as Layer


class Multilayernet:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation, weight_init_std, weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.5,
                 use_batchnormalization=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_batchnormalization = use_batchnormalization
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.params = {}

        self.__init_weight(weight_init_std)

        activation_layer = {'relu' : Layer.Relu, 'sigmoid' : Layer.Sigmoid}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num-1):
            self.layers['Affine' + str(idx)] = Layer.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

            if self.use_batchnormalization:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(self.hidden_size_list[idx-1])
                self.layers['Batchnorm' + str(idx)] = BatchNormalization

            self.layers['Activation' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Layer.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.Last_layer = Layer.SoftmaxWithLoss()


    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for key, layer in self.layers.items():
            x = layer.forward()

        return x
    
    def loss(self, x, t):
        y = self.predict(x)

        