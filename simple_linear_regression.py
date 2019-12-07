import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# split dataset into training set and test set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# import the dataset
dataset = pd.read_csv("Salary_Data.csv")

# independent variables
x = dataset.iloc[:, :-1].values
# dependent variables
y = dataset.iloc[:, 1].values


# Splitting dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature Scaling - no need on this one

# Fitting simple linear Regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)




