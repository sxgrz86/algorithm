import numpy as np
import math
import collections
from heapq import *
import copy
import random

def test(nums):
    arr = copy.deepcopy(nums)
    n = len(arr)
    _len = n
    for i in range(n):
        randidx = np.random.randint(0, _len)
        arr[randidx], arr[_len - 1] = arr[_len - 1], arr[randidx]
        _len -= 1
    return arr

nums = [i for i in range(10)]
for i in range(10):
    print(random.randint(0,1))
print(test(nums))



