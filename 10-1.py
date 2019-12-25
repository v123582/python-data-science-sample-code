# coding: utf-8

# # 铁达尼号

# 铁达尼号沉没是经典的沉船事件之一，在1912 年 的首航中，铁达尼号与冰山相撞后沉没，在2224名乘客和机组人员中造成1502人死亡。这场悲剧震惊了国际社会，之后并为船舶制定了更好的安全规定。造成海难失事的原因之一是乘客和机组人员没有足够的救生艇。尽管幸存者中存在运气因素，但有些人比其他人更容易生存，例如妇女，儿童和上流社会。
#

# ## 1.1 使用数据集与背景

# In[1]:



# 引入相关模块
import pandas as pd
import numpy as np

# 载入训练与测试数据
train_df = pd.read_csv('./data/Titanic_train.csv')
test_df = pd.read_csv('./data/Titanic_test.csv')


# ## 1.2 定义问题与观察数据

# ### 快速检阅数据样貌

# In[2]:


# 快速检阅数据样貌
print(train_df.columns.values)# 检视可以使用的栏位
# ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
# 'Ticket' 'Fare' 'Cabin' 'Embarked']
print(train_df.shape)# 检视数据数量
# (891, 12)
print(test_df.shape)# 检视数据数量
# (418, 11)


# In[3]:


# 检视训练数据前五笔
train_df.head()


# In[4]:


# 检视训练数据前五笔
train_df.tail()


# ### 定义数据栏位类型

# In[5]:


train_df.info()


# ### 确认类别数据分布

# In[6]:


# 确认类别数据分布
train_df.describe(include=['O'])


# In[7]:


test_df.describe(include=['O'])


# 从上图的结果当中，我们可以得到几个关于离散数据的观察：
#
#
# - 每个人姓名都不一样，没有重复
# - 男性乘客占比大约65%左右
# - 船票号码大约有23%是重复的，代表有许多人共用同一张船票
# - 不同客舱只有147间，代表许多乘客是共用客舱
# - 登船口共有三个，大多数乘客使用S区域登船口上船，大约占72%左右

# ### 确认数值数据范围

# In[8]:


train_df.describe()


# 从上图的结果当中，我们可以得到几个关于连续数据的观察：
#
#
# - 训练数据样本数共891笔，大概是泰坦尼克号船上总人数2,224的40％。
# - 从SibSp栏位可以看出大约75%乘客没有与家人一起乘船，另25％的乘客则有。
# - Fare栏位显示出票价差异很大，变异数达49多，且少数乘客（<1％）票价高达512美元。
# - Age栏位看出大部分乘客都很年轻平均29岁，且38岁以上的乘客较少。

# ### 比较数据间的关系

# In[9]:


train_df.corr()


# ## 1.3 数据清理与型态转换

# ### 先确认缺失值的状况

# In[10]:


train_df.info()


# In[11]:


test_df.info()


# In[12]:


train_df.isnull().sum()


# In[13]:


test_df.isnull().sum()


# ### 数据清理与型态转换

# In[14]:


# 性别字串转离散型数值
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
test_df['Sex'] = test_df['Sex'].map({'female': 1, 'male': 0}).astype(int)


# In[15]:


# Embarked登船口字串转离散型数值

## Embarked栏位有缺值，由前面数据统计结果得知只有少量 2 笔Embarked数据缺值，因此以众数进行补值
embarked_mode = train_df.Embarked.dropna().mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)

## 转离散值
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[16]:


# 「Ticket（船票）」和「Ticket（船票）」移除

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)


# In[17]:


# 用不同族群(Pclass/Sex)平均值来进行年龄(age)补值

for df in [train_df,test_df]:
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'Age'] = df[(df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
    df['Age'] = df['Age'].astype(int)


# In[18]:


# 对「Fare （费用）」栏位补中位数

test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].dropna().median())


# ### 检查处理后的结果

# In[19]:


train_df.info()


# In[20]:


test_df.info()


# In[21]:


train_df.isnull().sum()


# In[22]:


test_df.isnull().sum()


# ## 1.4 数据探索与视觉化

# In[23]:


train_df.corr()


# 从结果中，我们可以得知：
#
# - Parch、Fare、Embarked 三个栏位与是否可以生存是正相关，其中 Fare 相关性最强
# - PassengerId、Pclass、Age、SibSp 四个栏位与是否可以生存是负相关，其中 Pclass 有最强负相关
# - SibSp 和 Parch 特征具有零相关性值
#

# In[24]:


pd.pivot_table(train_df[['Survived','Pclass']], index=['Pclass'], aggfunc=np.mean)


# In[25]:


pd.pivot_table(train_df[['Survived','Sex']], index=['Sex'], aggfunc=np.mean)


# 从数据观察到 Pclass = 1 和Survived之间存在显著的相关性（> 0.5），建议在分类模型中包含此特征值。性别为女性的存活率非常高，表示获救的大部分是女性族群，此外从登船口 Cherbourg 进入的人生存率较高。
#

# ## 1.5 特征工程

# 由前面数据统计发现不同性别和不同年龄层的生存率差异大，因此可以藉由统计人名称呼来检视是否有其他的相关联。

# In[26]:



for dataset in [train_df,test_df]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in [train_df,test_df]:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', ' Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# 称乎字串转离散型数值
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in [train_df,test_df]:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[27]:


train_df.head()


# In[28]:


train_df.info()


# 「Name 」栏位与「PassengerID」都是流水号性质，对于分类任务不具区别性。 「Parch」、「SibSp」都具有与生存率相关性0的值，对于分类任务不具区别性。因此，这些栏位我们都不在模型中做使用：
#

# In[29]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name', 'PassengerId'], axis=1)
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)


# 接下来我们撷取不同年龄层范围，依照并将年龄层分成不同的范围，将连续转为离散型数值：

# In[30]:


for dataset in [train_df,test_df]:
    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 30), 'Age']= 1
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 45), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 60), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 4

train_df.head()


# Fare 栏位也采去一样的手法，由前面统计结果知道票价级距跟生存率关联高，故将票价依照四分位做分群：

# In[31]:


for dataset in [train_df,test_df]:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    
train_df.head()


# ## 1.6 机器学习

# In[32]:


X_train = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title']]
Y_train = train_df['Survived']
X_test = test_df


# In[33]:


# machine learning
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[34]:



# logistic regression
log = LogisticRegression(random_state=0)
log.fit(X_train, Y_train)
Y_pred = log.predict(X_test)
acc_log = round(log.score(X_train, Y_train) * 100, 2)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)


# Gaussian Naive Baye
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)


# In[35]:



models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Naive Bayes', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_gaussian, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)