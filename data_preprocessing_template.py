import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# for missing data 
from sklearn.preprocessing import Imputer



# import the dataset
dataset = pd.read_csv("Data.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# taking care of missing data
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


