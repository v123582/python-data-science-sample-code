# Pandas
import pandas as pd
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 'two' : pd.Series(['a', 'b', 'c', 'd'], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df.two = pd.Categorical(df.two).codes
df.two = pd.Categorical(df.two, categories=['a', 'b', 'c', 'd'],).codes

# scikit-learn
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.two)
df.two = le.transform(df.two)