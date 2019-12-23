import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# split dataset into training set and test set
from sklearn.model_selection import train_test_split
# feature scaling
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
# Backward elimination
import statsmodels.formula.api as smf


# import the dataset
dataset = pd.read_csv("50_Startups.csv")

# undependent variables
x = dataset.iloc[:, :-1].values
# dependent variables
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the dummy variable trap
x = x[:, 1:]

# Splitting dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 0)

# Fitting multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# Predicting the test set results
y_pred = regressor.predict(x_test)


# Building the optimal model using backward elimination






