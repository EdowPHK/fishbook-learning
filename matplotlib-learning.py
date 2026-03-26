import matplotlib.pyplot as plt
import numpy as np

# 1
x = np.arange(0, 6, 0.1)
y_1 = np.sin(x)
plt.plot(x, y_1)
# plt.show()

# 2
y_2 = np.cos(x)
plt.plot(x, y_1, label="sin")
plt.plot(x, y_2, linestyle="dashed",label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
# plt.show()

# 3
from matplotlib.image import imread
img = imread("cat.png")
plt.imshow(img)
# plt.show()