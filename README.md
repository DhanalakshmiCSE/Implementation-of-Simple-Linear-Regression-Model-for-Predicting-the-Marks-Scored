# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Dhanalakshmi
RegisterNumber:  212222040033

import pandas as pd
df= pd.read_csv('/content/student_scores.csv')
df.info()

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)



from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred =reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="silver")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_test,reg.predict(x_test),color="red")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()


mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)


a=np.array([[10]])
y_pred1=reg.predict(a)
print(y_pred1)

*/
```

## Output:
## 1. df.head():

![Screenshot 2023-10-17 151230](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/44991680-04c7-4f68-9951-212aef9af494)

## 2. df.tail():

![Screenshot 2023-10-17 151254](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/508f6ef6-13a6-4363-b537-cb1149ff769c)

## 3.Array value of X:

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/dae34db1-36e8-4b02-a09b-a26992db73ea)

## 4.Array value of Y:

![Screenshot 2023-10-17 151843](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/d0bffb42-101b-49f8-a1d6-ed64a705babc)

## 5.Values of Y prediction:

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/7a3d444b-3400-446a-9182-b4594afa2679)

##  6.Array values of Y test:

![Screenshot 2023-10-17 160227](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/cb6166b7-3609-4777-896e-86f540a7e3bb)

## 7.Training test graph:

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/6522149c-1337-464f-81c9-9cf72f338e5f)

## 8.Test set graph:

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/57ab15d8-15c3-4e89-a0a8-4d2114573a76)


## 9.Values of MSE,MAE,RMSE:

![image](https://github.com/DhanalakshmiCSE/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477832/6e2dd2dc-a79d-4f33-985b-538d33868a38)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
