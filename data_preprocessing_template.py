import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# for missing data 
from sklearn.preprocessing import Imputer
# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# import the dataset
dataset = pd.read_csv("Data.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# taking care of missing data
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


# Encoding categorical data
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

# dummy variables
# convert labels to multible binary columns
one_hot_encoder = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder.fit_transform(x).toarray()

# no need for dummy variables cuz its only 2 category and it will be 0,1
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)





