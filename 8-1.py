import pandas as pd

ipl_data = {
    'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
    'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
    'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
    'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
    'Points':[876,789,863,673,741,812,756,788,694,701,804,690]
}

df = pd.DataFrame(ipl_data)
#     Points  Rank    Team  Year
# 0      876     1  Riders  2014
# 1      789     2  Riders  2015
# 2      863     2  Devils  2014
# 3      673     3  Devils  2015
# 4      741     3   Kings  2014


df.groupby('Year').get_group(2014)
df.groupby('Year').agg(np.mean)
df.groupby('Year').agg([np.sum, np.mean, np.std])
df.groupby('Year').transform(np.mean)