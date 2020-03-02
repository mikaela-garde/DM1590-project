# documentation to pandas: https://pandas.pydata.org/pandas-docs/stable/index.html
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
import numpy as np

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
    if col_name != 'vote_average':
        col_num.append(col_name)

# Convension of data
# Encode the categorical columns
for col in col_cat:
    l_e = LabelEncoder()
    dt = l_e.fit_transform(df[col])
    df[col] = dt

vote_average = df['vote_average']
df = df.drop(columns=['vote_average'])


# ML- algorithm Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(df, vote_average, random_state=1, test_size=0.8)
gaus_model = GaussianNB()
h = np.array(X_train[col_num])
e = np.array(y_train)
gaus_model.fit(h, e)
#gaus_y_pred = gaus_model.predict(X_test[col_num])
#print(gaus_y_pred)

#cat_model = CategoricalNB()
#cat_model.fit(X_train[col_cat], y_train)
#cat_y_pred = cat_model.predict(X_test[col_cat])
#print(cat_y_pred)

