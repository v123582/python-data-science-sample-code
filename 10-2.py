# coding: utf-8

# # 房价预测

# 如果我们想要解决回归分析的问题，像是想要买新房子的时候，通常会想买到便宜空间又大的房子。不知道该考虑什么因素来决定房子，且担心的因素常不是影响价格的最重要原因，如果知道哪些因素才是真正会影响价格，就能够挑到能负担的房子，因此我们会需要用各种与房价有关因素来观察过去的数据。
#

# ## 2.1 使用数据集与背景

# In[1]:


# 引入相关模块
import pandas as pd
import numpy as np

# 载入训练与测试数据
train_df = pd.read_csv('./data/house_train.csv')
test_df = pd.read_csv('./data/house_test.csv')


# ## 2.2 定义问题与观察数据

# ### 首先，先快速的检阅数据的样貌，像是有哪些栏位、数据的大小：

# In[2]:


print(train_df.columns.values)# 检视可以使用的栏位
print(train_df.shape)# 检视数据数量
print(test_df.shape)# 检视数据数量


# ### 或是把前几笔、后几笔数据印出来看看：

# In[3]:


train_df.head()


# In[4]:


train_df.tail()


# ### 接着，我们必须先确认数据的栏位与类型：

# In[5]:


train_df.info()


# In[6]:


test_df.info()


# ### 查看数据的栏位后，我们分别连续与离散的数据做点观察。连续型的数据我们在意他的范围：

# In[7]:


train_df.describe()


# In[8]:


test_df.describe()


# ### 离散型的数据我们在意分布：

# In[9]:


train_df.describe(include=['O'])


# In[10]:


test_df.describe(include=['O'])


# ## 2.3 数据清理与型态转换

# ### 先确认缺失值的状况

# In[11]:


train_df.isnull().sum()


# In[12]:


test_df.isnull().sum()


# ### Utilities 栏位大部分值为 AllPub ，因无鉴别性，故删除此栏位：

# In[13]:


train_df.Utilities.value_counts()


# In[14]:


train_df = train_df.drop(['Utilities'], axis=1)
test_df = test_df.drop(['Utilities'], axis=1)


# ### 依照数据说明，以下含有 NA 栏位代表无相对应的值，可补值为通用值 None：

# In[15]:



train_df["PoolQC"].fillna("None",inplace=True)
test_df["PoolQC"].fillna("None",inplace=True)

train_df["MiscFeature"].fillna("None",inplace=True)
test_df["MiscFeature"].fillna("None",inplace=True)

train_df["Alley"].fillna("None",inplace=True)
test_df["Alley"].fillna("None",inplace=True)

train_df["Fence"].fillna("None",inplace=True)
test_df["Fence"].fillna("None",inplace=True)

train_df["FireplaceQu"].fillna("None",inplace=True)
test_df["FireplaceQu"].fillna("None",inplace=True)

train_df["GarageType"].fillna("None",inplace=True)
test_df["GarageType"].fillna("None",inplace=True)

train_df["GarageFinish"].fillna("None",inplace=True)
test_df["GarageFinish"].fillna("None",inplace=True)

train_df["GarageQual"].fillna("None",inplace=True)
test_df["GarageQual"].fillna("None",inplace=True)

train_df["BsmtQual"].fillna("None",inplace=True)
test_df["BsmtQual"].fillna("None",inplace=True)
    
train_df["BsmtCond"].fillna("None",inplace=True)
test_df["BsmtCond"].fillna("None",inplace=True)

train_df["GarageCond"].fillna("None",inplace=True)
test_df["GarageCond"].fillna("None",inplace=True)

train_df["BsmtExposure"].fillna("None",inplace=True)
test_df["BsmtExposure"].fillna("None",inplace=True)

train_df["BsmtFinType1"].fillna("None",inplace=True)
test_df["BsmtFinType1"].fillna("None",inplace=True)

train_df["BsmtFinType2"].fillna("None",inplace=True)
test_df["BsmtFinType2"].fillna("None",inplace=True)

train_df["Functional"].fillna("None",inplace=True)
test_df["Functional"].fillna("None",inplace=True)


# ### 下面这些栏位跟是跟地下室有关的，缺值代表可能代表没有地下室，数值型以 0 补值：

# In[16]:



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train_df[col].fillna(0,inplace=True)
    test_df[col].fillna(0,inplace=True)


# ### 相同 Neighborhood 的房子有可能有相似的 LotFrontage，故以 Neighborhood 房子的LotFrontage来补值：

# In[17]:



all_data = pd.concat([train_df,test_df],0)
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
train_df = all_data.iloc[:len(train_df),:]
test_df = all_data.iloc[len(train_df):,:]


# ### GarageYrBlt、GarageArea、GarageCars缺失值代表没有车子和车库以常数补值：

# In[18]:



train_df["GarageYrBlt"].fillna(-1,inplace=True)
test_df["GarageYrBlt"].fillna(-1,inplace=True)

train_df["GarageArea"].fillna(0,inplace=True)
test_df["GarageArea"].fillna(0,inplace=True)

train_df["GarageCars"].fillna(0,inplace=True)
test_df["GarageCars"].fillna(0,inplace=True)


# ### MasVnrType、MasVnrArea缺值代表墙壁可能没有装饰，数值型以0补值，离散型以none补值：

# In[19]:



train_df["MasVnrType"].fillna("None",inplace=True)
test_df["MasVnrType"].fillna("None",inplace=True)

train_df["MasVnrArea"].fillna(0,inplace=True)
test_df["MasVnrArea"].fillna(0,inplace=True)


# ### MSSubClass缺值代表无房子类型资讯，以none代替：

# In[20]:



train_df["MSSubClass"].fillna(0,inplace=True)
test_df["MSSubClass"].fillna(0,inplace=True)


# ### 最后几个栏位，我们利用统计方法进行补值：

# In[21]:



# 补众数
MSZoning_mode = train_df.MSZoning.mode()[0]
train_df["MSZoning"].fillna(MSZoning_mode,inplace=True)
test_df["MSZoning"].fillna(MSZoning_mode,inplace=True)

#以下特征只有一笔缺失值数据，以众数补值
for col in ['Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType']:
    mode = train_df[col].mode()[0]
    train_df[col].fillna(mode,inplace=True)
    test_df[col].fillna(mode,inplace=True)


# ### 最后一步是数据型态的调整，将有序类别字串转数值（标签编码）：

# In[22]:



from sklearn.preprocessing import LabelEncoder

all_data = pd.concat([train_df,test_df],0)
for col in ['MSSubClass', 'OverallCond','YrSold', 'MoSold']:
    all_data[col] = all_data[col].apply(str)
    
order_cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope','LotShape', 'PavedDrive' , 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')

for col in order_cols:
    le = LabelEncoder()
    le.fit(list(all_data[col].values))
    all_data[col] = le.transform(list(all_data[col].values))
    
train_df = all_data.iloc[:len(train_df),:]
test_df = all_data.iloc[len(train_df):,:]


# In[23]:


train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)


# ### 检查处理后的结果

# In[24]:


train_df.isnull().sum()


# In[25]:


test_df.isnull().sum()


# ## 2.4 数据探索与视觉化

# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = train_df.corr()
cols = correlation_matrix.nlargest(10,'SalePrice')['SalePrice'].index
correlation_matrix = train_df[cols].corr()
plt.figure(figsize = (12,16))
sns.heatmap(correlation_matrix,annot=True,xticklabels = cols.values ,annot_kws = {'size':20},yticklabels = cols.values)


# ## 2.5 特征工程

# In[27]:



from scipy.special import boxcox1p
from scipy import stats
#将非正态特征转正态
all_data = pd.concat([train_df,test_df],0)
skew = all_data.skew()
skew = skew[abs(skew) > 1]
skew_features = skew.index
# skew_features = skew_features.drop("SalePrice")

for feat in skew_features:
    all_data[feat] = boxcox1p(all_data[feat], 0.2)

train_df = all_data.iloc[:len(train_df),:]
test_df = all_data.iloc[len(train_df):,:]
#将SalePrice做log normalization
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])


# ## 2.6 机器学习

# In[28]:



from sklearn.model_selection import train_test_split
cols = set(test_df.columns)
X = train_df[list(cols)]
y = train_df['SalePrice']
X_test = test_df[list(cols)]
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.33, random_state=42)


# In[29]:


train_df[list(cols)].isnull().any()


# In[30]:


train_df['SalePrice'].isnull().any()


# In[31]:


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge

def apply_classify(model):
    model.fit(X_train, y_train)
    Y_dev = model.predict(X_dev)
    rmse = mean_squared_error(y_dev, Y_dev)
    return rmse

classifier = { 'linear regression':LinearRegression(),
               'Ridge': Ridge()
             }

result = {'classifier':[],'rmse':[]}
for name,c in classifier.items():
    rmse = apply_classify(c)
    result['classifier'].append(name)
    result['rmse'].append(rmse)


# In[32]:


result = pd.DataFrame(result)
result.sort_values(by='rmse', ascending=False)