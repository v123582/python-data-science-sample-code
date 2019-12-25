# Using Pandas
import pandas as pd 
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c']})
pd.get_dummies(df)

# Using sklearn
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
enc.transform([[0, 1, 3]]).toarray()