# documentation to pandas: https://pandas.pydata.org/pandas-docs/stable/index.html
import pandas as pd

data1 = pd.read_csv('tmdb_5000_credits.csv',sep=',')
data2 = pd.read_csv('tmdb_5000_movies.csv',sep=',')

data = pd.merge(data1, data2)

# DATA SELECTION
# droping data that is not relevant
df = data.copy()

# Kanske att man ska ta bort crew och id också
df = df.drop(columns=['movie_id', 'keywords', 'original_title', 'overview', 'status', 'tagline', 'title', 'homepage'])

# Droping the 3 rows with na-values
df = pd.DataFrame.dropna(df, axis=0)


# Splip numeric data och categorical data
categorical = df.select_dtypes(include=['object']).copy()
col_cat = []
for col_name in categorical.columns:
    col_cat.append(col_name)

numeric = df.select_dtypes(include=['float64', 'int64']).copy()
col_num = []
for col_name in numeric.columns:
    col_num.append(col_name)

#col_cat = ['cast', 'crew', 'genres', 'original_language']
#print(df.info())
#ML- algorithm Naive Bayes


