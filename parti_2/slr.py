# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 04:58:57 2022

@author: niku

"""

"""
Simple linear Regression :- 

1. is type of regression
2.this algorithem relationship between dependent variable and single independent variable.
3.A slped straight line, hence it is called simple Linear Regression.
4.is key point is that the dependent vasriable must be acontiouns\/real value.

5. SLR mainly two objective
1. model the relaionship between the two variable
2.Forecasting new observations.


6.SLR Equcations

y=a0+a1X+ e

a0- the intercept of the Regression line
a1- it is the slope of the regression line, which tells whether the line incresing and decresing.



# Problem Statements 

1.We want to find out if there is any correlation between these two variables
2.We will find the best fit line for the dataset.
3.How the dependent variable is changing by changing the independent variable.

# independent variable
1 EXPERIECE

#dependent variable
1.Salary


"""
# Data Pre-Processing.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

data_set= pd.read_csv("../Salary_Data.csv")


x = data_set.iloc[:,:-1].values # yearrs of eperence
y = data_set.iloc[:,1].values # Salray


"""
 split the both variable
 
 1. we have 30 observation
 2. we will take 20 observation training set 
 3. 10 observation for the test set
 
"""
# Spliting the dataset into traning and test set.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,_y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# step-2 Fitting the simple Linear REgression to the Training SET
"""
1. importLinear Regression class of the linear_model

2. from the scikit learn . after importing the class we are going to create an object
    of the class named as a REGRESSOR 

3. fit() passed the x_train and y_train 
   1. for wehich data set for the dependent and independent
   2. the model can easily lern the correlation between the predictor and target variable
"""
# Fitting the Simple Linear REgression model to the TRaining

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#. Prediction of Test and Trainign set result

y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)

# step 4 visualizing the Training set result


mtp.scatter(x_train,y_train,color="green")
mtp.plot(x_train,x_pred,color="red")






















