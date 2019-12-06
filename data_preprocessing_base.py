import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# for missing data 
from sklearn.preprocessing import Imputer
# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# split dataset into training set and test set
from sklearn.model_selection import train_test_split
# feature scaling
from sklearn.preprocessing import StandardScaler



# import the dataset
dataset = pd.read_csv("Data.csv")

# undependent variables
x = dataset.iloc[:, :-1].values
# dependent variables
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


# Splitting dataset into the training set and test set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 0)


# feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
# no need to fit test set cuz its allready fited in train set
x_test = sc_x.transform(x_test)







