import numpy as np
a = np.arange(15).reshape(3, 5)
print(a)
# array([[ 0, 1, 2, 3, 4],
# [ 5, 6, 7, 8, 9],
# [10, 11, 12, 13, 14]])

print(a.shape) # (3, 5)
print(a.ndim) # 2
print(a.itemsize) # 8
print(a.size) # 15
print(a.dtype) # 'int64'
print(type(a)) # <type 'numpy.ndarray'>
print(dir(a))


# 从 Python 的 List 中元素的类型推导出来的

a = np.array([2,3,4])

print(a) # array([2, 3, 4])
print(a.dtype) # int64
print(a.dtype.name) # int64
print(a.dtype == 'int64') # True
print(a.dtype is 'int64') # False
print(a.dtype is np.dtype('int64')) # True


b = np.array([1.2, 3.5, 5.1])

print(b) # [1.2 3.5 5.1]
print(b.dtype) # float64
print(b.dtype.name) # float64
print(b.dtype == 'float64') # True
print(b.dtype is 'float64') # False
print(b.dtype is np.dtype('float64')) # True
print(b == [1.2, 3.5, 5.1]) # False


c = np.array([(1.5,2,3), (4,5,6)])
d = np.array([[1,2], [3,4]], dtype=complex)
print(c)
print(c.dtype)
print(d)
print(d.dtype)