import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy as sp 
import statsmodels.api as sm

data = pd.read_csv('startups.csv')
real_x = data.iloc[:,0:4].values
real_y = data.iloc[:,4].values

le = LabelEncoder()
real_x[:,3] = le.fit_transform(real_x[:,3])
#print(real_x[:,3])

oneHE = OneHotEncoder(categorical_features=[3])
real_x = oneHE.fit_transform(real_x).toarray()
#print(real_x)

real_x = real_x[:,1:]
real_x = np.append(arr = np.ones((50,1)).astype(int),values = real_x, axis=1)

x_opt = real_x[:,[0,3]]

reg_OLS = sm.OLS(endog=real_y, exog=x_opt).fit()
print(reg_OLS.summary())

training_x,testing_x,training_y,testing_y = train_test_split(real_x,real_y,test_size = 0.2,random_state=0)
MLR = LinearRegression()
MLR.fit(training_x,training_y)
pred_y = MLR.predict(testing_x)

print(pred_y - testing_y)