#Simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the DataSet
dataset=pd.read_csv('Salary_data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values


# Splitting the dataSet into training and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

#Fitting Simple Linear REgression to the trainig set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the Test Set Results
y_pred=regressor.predict(x_test)

#Visualising the Training Set Results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Years of experience(Training set)')
plt.xlabel('years of experinece')
plt.ylabel('Salary')
plt.show()

#Visualising the Test Set Results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Years of experience(Training set)')
plt.xlabel('years of experinece')
plt.ylabel('Salary')
plt.show()