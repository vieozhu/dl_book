# coding: utf-8
from ch05.layer_naive import *

apple = 100  # apple单价
apple_num = 2  # apple个数
tax = 1.1  # 税率

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# print("apple_price:", int(apple_price))
# print("price:", int(price))
print("dprice:各输入变量加1，对总价的影响")
print("dTax:税率增加1(100%)总价上升(元)", dtax)
print("dApple:苹果价格增加1元总价上升(元)", dapple)
print("dApple_num:苹果增加1个总价上升(元)：", int(dapple_num))
