import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor

# t1 = Tensor([1, 2, 3, 4, 5])

# print(t1.numpy())

# full = Tensor.full(shape=(2, 3), fill_value=5)

# t4 = Tensor([1, 2, 3, 4, 5])
# t5 = (t4 + 1) * 2
# t6 = (t5 * t4).relu().log_softmax()
# print(t6.numpy())

class TinyNet:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x

net = TinyNet()
