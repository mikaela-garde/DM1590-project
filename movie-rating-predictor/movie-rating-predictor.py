# documentation to pandas: https://pandas.pydata.org/pandas-docs/stable/index.html
import pandas as pd

data1 = pd.read_csv('tmdb_5000_credits.csv',sep=',')
data2 = pd.read_csv('tmdb_5000_movies.csv',sep=',')
print(data1.isna())

