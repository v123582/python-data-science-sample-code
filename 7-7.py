
df = pd.DataFrame({'size':['XXL', 'XL', 'L', 'M', 'S']})

# Using Pandas
import pandas as pd 
cat = pd.Categorical(df['size'], categories=df['size'].unique(), ordered=True)
df['size_code'] = cat.codes

# Using sklearn
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
le.fit(df['size'])
le.transform(df['size'])