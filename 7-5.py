# 载入所需要的模块
import pandas as pd
import numpy as np

# 定义一个由 dict 所转成的 DataFrame
d = {
    'one' : pd.Series(['1', '1', '1.0'], index=['a', 'b', 'c']),
    'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(d)

# 以下操作可以根据需求使用位置或是名称作为筛选条件
print(df.ix[0, 'one']) # 数据
print(df.ix['a'][0]) # 数据

print(df.ix[0, ['one']]) # 向量
print(df.ix[['a'], 0]) # 向量

print(df.ix[['a'], [0]]) # DataFrame
print(df.ix[[0], :]) # DataFrame
print(df.ix[:, ['one']]) # DataFrame
print(df.ix[:, :]) # DataFrame