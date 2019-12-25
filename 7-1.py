# 载入所需要的模块
import pandas as pd
import numpy as np

# 定义一个由 dict 所转成的 DataFrame
d = {
  'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
  'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(d)

print(df.one) # 包含 one 栏位的 series
# a 1.0
# b 2.0
# c 3.0
# d NaN
# Name: one, dtype: float64

print(df['one']) # 包含 one 栏位的 series
# a 1.0
# b 2.0
# c 3.0
# d NaN
# Name: one, dtype: float64

print(df[['one']]) # 包含 one 栏位的 DataFrame
# one
# a 1.0
# b 2.0
# c 3.0
# d NaN