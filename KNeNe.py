import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

data = pd.read_csv('Ads.csv')
#print(data.head(10))

real_x = data.iloc[:,[2,3]].values
real_y = data.iloc[:,4].values

training_x,test_x,training_y,test_y = train_test_split(real_x,real_y,test_size = 0.25,random_state = 0)
#print(training_x)
s_c = StandardScaler()
training_x = s_c.fit_transform(training_x)
test_x = s_c.fit_transform(test_x)

KNN_cls = KNeighborsClassifier(n_neighbors = 5,metric ="minkowski",p=2)
KNN_cls.fit(training_x,training_y)

y_pred = KNN_cls.predict(test_x)
#print(y_pred)

c_mat = confusion_matrix(test_y,y_pred)
print(c_mat)

X_set, y_set = training_x, training_y
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01)
                   ,np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01))

plt.contourf(X1,X2,KNN_cls.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],c=ListedColormap(('red','green'))(i),label = j)

plt.show()

X_set, y_set = test_x, test_y
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01)
                   ,np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01))

plt.contourf(X1,X2,KNN_cls.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],c=ListedColormap(('red','green'))(i),label = j)

plt.show()