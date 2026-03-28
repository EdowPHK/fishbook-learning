import pickle

# 1. 序列化（把对象存成文件）
data = {"mnist_images": [1,2,3], "labels": [0,1,2]}     # 随便一个Python对象
with open("data.pkl", "wb") as f:                       # wb = 二进制写入
    pickle.dump(data, f)                                # 把data对象打包写入文件

# 2. 反序列化（从文件恢复对象）
with open("data.pkl", "rb") as f:                       # rb = 二进制读取
    loaded_data = pickle.load(f)                        # 完整恢复出原来的data对象

print(loaded_data)                                      # 输出和原来完全一样的字典