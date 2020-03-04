# documentation to pandas: https://pandas.pydata.org/pandas-docs/stable/index.html
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
import numpy as np
import ast
from collections import Counter

data1 = pd.read_csv('tmdb_5000_credits.csv',sep=',')
data2 = pd.read_csv('tmdb_5000_movies.csv',sep=',')

data = pd.merge(data1, data2)

# DATA SELECTION
# droping data that is not relevant
df = data.copy()

# Kanske att man ska ta bort crew och id också
df = df.drop(columns=['movie_id','keywords', 'original_title', 'overview', 'status', 'tagline', 'title', 'homepage'])

# Droping the 3 rows with na-values
df = pd.DataFrame.dropna(df, axis=0)


# Split numeric data och categorical data
categorical = df.select_dtypes(include=['object']).copy()
col_cat = []
for col_name in categorical.columns:
    col_cat.append(col_name)

numeric = df.select_dtypes(include=['float64', 'int64']).copy()
col_num = []
for col_name in numeric.columns:
    if col_name != 'vote_average':
        col_num.append(col_name)

columns_dicts = ['cast', 'crew', 'genre', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']


release_year = []
release_month_day = []
for date in df['release_date']:
    date = date.split("-")
    release_year.append(int(date[0]))
    release_month_day.append((date[1]+date[2]))

df['release_year'] = release_year
df['release_year'].astype(int)
df['release_month_day'] = release_month_day
df['release_month_day'].astype(int)
df.drop(columns=['release_date'])

# Get name of all actors
cast = data[columns_dicts[0]]
cast_names = []
for row in cast:
    f_lst = ast.literal_eval(row)
    for f_dict in f_lst:
        cast_names.append(f_dict.get("name"))

# Get 10 most common actors
most_common_names = []
most_common = Counter(cast_names).most_common(30)
for i in range(len(most_common)):
    most_common_names.append(most_common[i][0])

# Create new column for each actor
for cast_member in most_common:
    df[str(cast_member[0])] = np.zeros(df.shape[0])

#Set 1 on the movie with the actor
index = 0
for row in df['cast']:
    f_lst = ast.literal_eval(row)
    for person in f_lst:
        if person.get("name") in most_common_names:
            df.at[index, person.get("name")] = 1
    index += 1


# Convension of data
# Encode the categorical columns
for col in col_cat:
    l_e = LabelEncoder()
    dt = l_e.fit_transform(df[col])
    df[col] = dt

vote_score = []
for value in df['vote_average']:
    if value >= 6:
        vote_score.append(1)
    else:
        vote_score.append(0)
df = df.drop(columns=['vote_average'])


# ML- algorithm Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(df, vote_score, random_state=0, test_size=0.5)
gaus_model = GaussianNB()
gaus_model.fit(X_train[col_num], y_train)
y_pred_gaus = gaus_model.predict_log_proba(X_test[col_num])

cat_model = CategoricalNB()
cat_model.fit(X_train[col_cat], y_train)
#y_pred_cat = cat_model.predict_log_proba(X_test[col_cat])


y_pred = y_pred_gaus
#+ y_pred_cat
y_pred = np.argmax(y_pred, axis=1)
y_pred_sum = np.sum(y_pred == y_test)/len(y_test)

print("The performance of the Naive Bayes classifier is: ")
print(np.round(y_pred_sum*100, 2), "%")

