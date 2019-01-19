from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (Imputer, LabelEncoder, StandardScaler, MinMaxScaler)
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Importing Data
dataset = pd.read_csv('Data.csv')

# Removing some columns which aren't useful for our calculation
df = dataset.drop([col for col in ['movie_title', 'color', 'plot_keywords', 'movie_imdb_link',
                                   'aspect_ratio', 'genres', 'actor_2_name', 'actor_3_name', 'actor_1_name',
                                   'director_name']
                   if col in dataset], axis=1)

# get the positions of the columns which are strings
language_pos = df.columns.get_loc("language")
country_pos = df.columns.get_loc("country")
content_rating_pos = df.columns.get_loc("content_rating")

# create a exclude list of these excluded attributes
categorical_fts = []
categorical_fts.append(language_pos)
categorical_fts.append(country_pos)
categorical_fts.append(content_rating_pos)

# Array of features, exludes the last column
X = df.iloc[:, :-1].values

# Last column array, string of length 6 (dtype)
Ystr = np.asarray(df.iloc[:, df.shape[1] - 1], dtype="|S6")  # numpy is moody
# Convert it to float
Y = Ystr.astype(np.float)

# Keeps throwing the error msg
label_language = LabelEncoder()

X[0:, language_pos] = label_language.fit_transform(X[0:, language_pos].astype(np.str))
label_country = LabelEncoder()
X[0:, country_pos] = label_country.fit_transform(X[0:, country_pos].astype(np.str))
label_content_rating = LabelEncoder()
X[0:, content_rating_pos] = label_content_rating.fit_transform(X[0:, content_rating_pos].astype(np.str))

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = imp.fit_transform(X)

# Lets pick the important features
estimator = RandomForestClassifier()
# estimator = svm.SVR(kernel="linear")
rfe = RFE(estimator, 15)
rfe = rfe.fit(X, Ystr)

# print(rfe.support_)    #These are the coloumns we're keeping
# print(rfe.ranking_)    #These are ranking the coloumns
# Drop the unimportant features
drop_list = []
for i in range(0, len(rfe.support_)):
    # print(df.columns.values[i], " ranking:", rfe.ranking_[i])
    if not rfe.support_[i]:
    #     print(df.columns.values[i], " ranking: ", rfe.ranking_[i])  # TODO Remove this later
    # else:
        drop_list.append(i)
X = np.delete(X, drop_list, axis=1)
# Scaling X

"""
We don't use OneHotEncoder as we are already using LabelEncoder.
Whoever suggested we use OneHotEncoder was an idiot.
"""

# scaler = StandardScaler().fit(X)
scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)
test_movie = X[-4:]
X = X[0:len(X)-4]
test_res = Ystr[-4:]
Ystr = Ystr[0:len(Ystr)-4]

X_train, X_test, Y_train, Y_test = train_test_split(X, Ystr, test_size=0.15, random_state=0)
