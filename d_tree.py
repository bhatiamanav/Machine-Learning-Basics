import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor 

data = pd.read_csv('salaries_pos.csv')
real_x = data.iloc[:,1:2].values
real_y = data.iloc[:,2].values

dec_reg = DecisionTreeRegressor(random_state=0)
dec_reg.fit(real_x,real_y)

x_req = np.array(6.5).reshape(-1,1)
y_pred = dec_reg.predict(x_req)
print(y_pred)

x_grid = np.arange(min(real_x),max(real_x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(real_x,real_y,color='blue')
plt.plot(x_grid,dec_reg.predict(x_grid),color='red')
plt.show()