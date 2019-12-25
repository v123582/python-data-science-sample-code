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


pd.pivot_table(df, values='Points', index=['Year'], columns=['Team'])
# Team  Devils  Kings  Riders  Royals  kings
# Year
# 2014   863.0  741.0   876.0   701.0    NaN
# 2015   673.0    NaN   789.0   804.0  812.0
# 2016     NaN  756.0   694.0     NaN    NaN
# 2017     NaN  788.0   690.0     NaN    NaN


pd.pivot_table(df, values='Points', index=['Year'], columns=['Team'], aggfunc=len)
# Team  Devils  Kings  Riders  Royals  kings
# Year
# 2014     1.0    1.0     1.0     1.0    NaN
# 2015     1.0    NaN     1.0     1.0    1.0
# 2016     NaN    1.0     1.0     NaN    NaN
# 2017     NaN    1.0     1.0     NaN    NaN


pd.crosstab(df.Year, df.Team)
# Team  Devils  Kings  Riders  Royals  kings
# Year
# 2014       1      1       1       1      0
# 2015       1      0       1       1      1
# 2016       0      1       1       0      0
# 2017       0      1       1       0      0