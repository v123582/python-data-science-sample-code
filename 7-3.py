# 载入所需要的模块
import pandas as pd
import numpy as np

# 定义一个由 dict 所转成的 DataFrame
d = {
    'one' : pd.Series(['1', '1', '1.0'], index=['a', 'b', 'c']),
    'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(d)

# 取出一个数据数值
print(df.loc['a', 'one']) # 数据
print(df.loc['a']['one']) # 数据

# 取出包含一个以 row 为底的，包含 column 数据的向量
print(df.loc['a', ['one']]) # 向量
print(df.loc['a', ['one', 'two']]) # 向量
print(df.loc['a', 'one':'two']) # 向量
print(df.loc['a', :]) # 向量
print(df.loc['a']) # 向量

# 取出包含一个以 column 为底的，包含 row 数据的向量
print(df.loc[['a'], 'one']) # 向量
print(df.loc[['a', 'b', 'c', 'd'], 'one']) # 向量
print(df.loc['a':'d', 'one']) # 向量
print(df.loc[:, 'one']) # 向量

# 取出一个子 DataFrame
print(df.loc[['a'], ['one']]) # DataFrame
print(df.loc[['a'], :]) # DataFrame
print(df.loc[:, ['one']]) # DataFrame
print(df.loc[:, :]) # DataFrame