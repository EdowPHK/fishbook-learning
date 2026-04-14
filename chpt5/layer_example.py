
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        # 此处不绑定具体数值(def __init__(self, x, y):self.x = x, self.y = y)以保证复用性。

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

apple = 100
orange = 150
apple_num = 2
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
add_apple_orange = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(add_apple_orange, tax)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print(dapple_num, dapple, dorange_num, dorange, dtax)