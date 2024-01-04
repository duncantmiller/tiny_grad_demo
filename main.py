import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor

t1 = Tensor([1, 2, 3, 4, 5])

print(t1.numpy())

full = Tensor.full(shape=(2, 3), fill_value=5)

t4 = Tensor([1, 2, 3, 4, 5])
t5 = (t4 + 1) * 2
t6 = (t5 * t4).relu().log_softmax()
print(t6.numpy())
