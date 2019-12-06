import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# split dataset into training set and test set
from sklearn.model_selection import train_test_split
# feature scaling
# from sklearn.preprocessing import StandardScaler



# import the dataset
dataset = pd.read_csv("Data.csv")

# undependent variables
x = dataset.iloc[:, :-1].values
# dependent variables
y = dataset.iloc[:, 3].values


# Splitting dataset into the training set and test set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 0)


# feature scaling
# sc_x = StandardScaler()
# x_train = sc_x.fit_transform(x_train)
# no need to fit test set cuz its allready fited in train set
# x_test = sc_x.transform(x_test)







