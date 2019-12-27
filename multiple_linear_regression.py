import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# split dataset into training set and test set
from sklearn.model_selection import train_test_split
# feature scaling
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
# Backward elimination
import statsmodels.formula.api as smf
import statsmodels.api as sm


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
# onehotencoder = OneHotEncoder(categorical_features = [3])
ct = ColumnTransformer(
    # The column numbers to be transformed (here is [3])
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    # Leave the rest of the columns untouched
    remainder='passthrough'
)
# x = onehotencoder.fit_transform(x).toarray()
x = np.array(ct.fit_transform(x), dtype=np.float)

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
x = np.append(np.ones((50, 1)).astype(int), x, axis=1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()
# removed 2 cuz it has highest p value that is greater then significant level
x = np.append(np.ones((50, 1)).astype(int), x, axis=1)
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()
# removed 1 cuz it has highest p value that is greater then significant level
x = np.append(np.ones((50, 1)).astype(int), x, axis=1)
x_opt = x[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()
# removed 4 cuz it has highest p value that is greater then significant level
x = np.append(np.ones((50, 1)).astype(int), x, axis=1)
x_opt = x[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()
# removed 5 cuz it has highest p value that is greater then significant level
x = np.append(np.ones((50, 1)).astype(int), x, axis=1)
x_opt = x[:, [0, 3]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
regressor_ols.summary()


