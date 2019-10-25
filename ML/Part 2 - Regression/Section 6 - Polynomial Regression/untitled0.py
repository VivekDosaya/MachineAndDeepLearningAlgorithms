# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)


#Friitng Polynomial regressions to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4) #start trying degree 2 and then start building up
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

#Visualising Linear REgression Model
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show

#Visualisinhg Polynomial Regression Model
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue') #not x_poly but use predict instead as x_poly is already defined
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show

#for having an advanced plot ie levels will be 0-0.1 instead of 0-1
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color='blue') #not x_poly but use predict instead as x_poly is already defined
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)
#Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform(6.5))







