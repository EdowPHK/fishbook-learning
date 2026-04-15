# 最优化方法(Use Ctrl+Shift+V to preview)

## 1. SGD（Stochastic Gradient Descent，随机梯度下降）

### 1.1 基本思想
每一步都沿着当前参数点的负梯度方向更新参数：

$$
	heta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

- $\theta_t$：第 $t$ 次迭代的参数
- $\eta$：学习率（learning rate）
- $L(\theta)$：损失函数

### 1.2 直觉理解
可以把 SGD 想成“每次只看当前坡度，然后往下走一小步”。

- 优点：实现简单，计算开销小
- 缺点：在狭长谷底（非均向曲面）会来回震荡，收敛路径低效

### 1.3 伪代码
```python
# theta: 参数
# grad(theta): 当前梯度
theta = theta - lr * grad(theta)
```

---

## 2. Momentum（动量法）

### 2.1 基本思想
Momentum 在 SGD 基础上引入“速度”变量，累计历史梯度信息：

$$
v_{t+1} = \alpha v_t - \eta \nabla_\theta L(\theta_t)
$$

$$
	heta_{t+1} = \theta_t + v_{t+1}
$$

- $v_t$：速度（动量项）
- $\alpha$：动量系数（常用 $0.9$）
- $\eta$：学习率

### 2.2 直觉理解
像一个带惯性的球在损失曲面上滚动：

- 在“方向一致”的梯度上会加速前进
- 在“来回变化”的方向上会互相抵消一部分震荡

因此相比 SGD，Momentum 往往在狭长谷地中收敛更快、更稳定。

### 2.3 伪代码
```python
# 初始化
v = 0

# 每一步
v = alpha * v - lr * grad(theta)
theta = theta + v
```

---

## 3. AdaGrad（自适应学习率）

### 3.1 基本思想
AdaGrad 会为每个参数维护历史梯度平方和，并按参数维度自适应地缩放学习率：

$$
h_{t+1} = h_t + g_t \odot g_t
$$

$$
	heta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{h_{t+1}} + \epsilon}
$$

- $g_t = \nabla_\theta L(\theta_t)$：当前梯度
- $h_t$：历史梯度平方累积
- $\epsilon$：防止分母为 0 的小常数（常用 $10^{-7}$）

### 3.2 直觉理解
可以理解为“经常变化的方向，步长会自动变小；变化少的方向，步长相对更大”。

- 优点：对稀疏特征友好，初期收敛通常较快
- 缺点：$h_t$ 持续累积，后期学习率可能衰减过快，导致训练变慢

### 3.3 伪代码
```python
# 初始化
h = 0

# 每一步
g = grad(theta)
h = h + g * g
theta = theta - lr * g / (np.sqrt(h) + 1e-7)
```

---

## 4. Adam（Adaptive Moment Estimation）

### 4.1 基本思想
Adam 结合了 Momentum（一阶矩）和 RMSProp 思路（二阶矩）：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) (g_t \odot g_t)
$$

再做偏差修正：

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

参数更新：

$$
	heta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

- $m_t$：一阶矩估计（梯度均值）
- $v_t$：二阶矩估计（梯度平方均值）
- 常用超参数：$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$

### 4.2 直觉理解
Adam 既有动量带来的“方向稳定性”，又有自适应学习率带来的“尺度适配性”。

- 优点：通常开箱即用，收敛速度快，对超参数不太敏感
- 缺点：在部分任务上的泛化能力不一定优于 SGD/Momentum

### 4.3 伪代码
```python
# 初始化
m, v, t = 0, 0, 0

# 每一步
t = t + 1
g = grad(theta)
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * (g * g)
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
theta = theta - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
```

---