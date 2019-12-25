# 载入所需要的模块
import pandas as pd
import numpy as np

# 定义一个由 dict 所转成的 DataFrame
d = {
    'one' : pd.Series(['1', '1', '1.0'], index=['a', 'b', 'c']),
    'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(d)

print(df[0:1]) # 包含 第 0 到 第 0 列数据的 dataframe
# one two
# a 1 1

print(df[0:2])# 包含 第 0 到 第 1 列数据的 dataframe
# one two
# a 1 1
# b 1 2