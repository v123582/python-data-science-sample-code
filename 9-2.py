# Pandas
import pandas as pd
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c']})
df
#    A  B  
# 0  a  b  
# 1  b  a  
# 2  a  c  
pd.get_dummies(df)
#    A_a  A_b  B_a  B_b  B_c
# 0    1    0    0    1    0
# 1    0    1    1    0    0
# 2    1    0    0    0    1

# scikit-learn
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.B)
df.B = le.transform(df.B)                   
print(df)
enc = preprocessing.OneHotEncoder()
enc.fit(df)
print(pd.DataFrame(enc.transform(df).toarray()))