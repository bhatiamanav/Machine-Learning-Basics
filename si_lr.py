import numpy as numpy
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

req_data = pd.read_csv('salary.csv')
real_x = req_data.iloc[:,0].values
real_y = req_data.iloc[:,1].values
real_x = real_x.reshape(-1,1)
real_y = real_y.reshape(-1,1)

training_x,testing_x,training_y,testing_y = train_test_split(real_x,real_y,test_size = 0.6,random_state=0)
NLR = LinearRegression()
NLR.fit(training_x,training_y)
pred_y = NLR.predict(testing_x)

print(pred_y)
print(testing_y)

plt.scatter(testing_x,testing_y)
plt.plot(testing_x,pred_y)
plt.show()


