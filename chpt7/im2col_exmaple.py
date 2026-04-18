import sys, os
sys.path.append(os.getcwd())
import numpy as np
from fsbook_code.common.util import im2col

# 返回的2维数组(x, y): y —— 滤波器元素总和
x1 = np.random.randn(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, 1, 0)
print(col1.shape)

x2 = np.random.randn(10, 3, 7, 7)
col2 = im2col(x2, 5, 5 , 1, 0)
print(col2.shape)

