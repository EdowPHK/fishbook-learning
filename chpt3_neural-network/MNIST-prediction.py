from mnist import load_mnist
from PIL import Image
import numpy as np
import pickle
from neural_network import sigmoid,softmax

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)   # \：换行续写符
    return x_test, t_test

def init_network():
    with open("chpt3_neural-network/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)    # 沿着第一维的方向获取概率最高的元素的索引
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    # flatten=True时读入的图像是以一列（一维）Numpy数组的形式保存的，显示这个图象时需要变为原来的28*28像素的形状
    # print(x_train.shape)
    img = x_train[0]
    label = t_train[0]
    # print(label)
    # print(img.shape)
    img = img.reshape(28, 28)
    # print(img.shape)
    # img_show(img)
