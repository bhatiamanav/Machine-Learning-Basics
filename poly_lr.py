import numpy as numpy
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('salaries_pos.csv')
real_x = data.iloc[:,1:2].values
real_y = data.iloc[:,2].values

LR = LinearRegression()
LR.fit(real_x,real_y)

PLR = PolynomialFeatures(degree=5)
real_x_poly = PLR.fit_transform(real_x)
LR2 = LinearRegression()
LR2.fit(real_x_poly,real_y)

plt.scatter(real_x,real_y,color='red')
plt.plot(real_x,LR.predict(real_x),color = 'blue')
plt.plot(real_x,LR2.predict(real_x_poly),color = 'green')
plt.show()