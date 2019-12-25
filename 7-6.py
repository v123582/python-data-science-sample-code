# 载入所需要的模块
import pandas as pd
import numpy as np

# 定义一个由 dict 所转成的 DataFrame
d = {
    'one' : pd.Series(['1', '1', '1.0'], index=['a', 'b', 'c']),
    'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
}
df = pd.DataFrame(d)

# 产生一个 df == '1' 的 Boolean DataFrame
print(df == '1')
# one two
# a True False
# b True False
# c False False
# d False False

# 利用 Boolean DataFrame 作为筛选条件
print(df[df == '1'])
# one two
# a 1 NaN
# b 1 NaN
# c NaN NaN
# d NaN NaN

# 产生一个 df.one == '1' 的 Boolean DataFrame
print(df[df.one == '1'])
# one two
# a 1 1
# b 1 2

# 利用 Boolean DataFrame 作为筛选条件
print(df.loc[df.one == '1'])
# one two
# a 1 1
# b 1 2