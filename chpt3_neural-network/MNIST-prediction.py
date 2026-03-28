from mnist import load_mnist
from PIL import Image
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

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