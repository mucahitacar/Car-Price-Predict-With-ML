# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 00:13:25 2020

@author: mucah
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
 
"""
x=dataframe.iloc[:, :-1].values
y=dataframe.iloc[:, 6].values

"""
#linear_regressor.predict([[2016,8000,0,1,0,0,0,0,1]])

dataframe=pd.read_csv("sayili.csv")
x=dataframe.drop('fiyat',axis='columns')
y=dataframe.fiyat



from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(x,y)

y_predictions = linear_regressor.predict(x)

hata=y_predictions-y

m=len(y)
numb=0
for i in range(m):
    
    numb +=abs(y_predictions[i]-y[i])

toplam=round(numb/m,1)




m=len(y)
numb=0
for i in range(m):
    if abs(y_predictions[i]-y[i])>100000:
     numb=numb+1 
    




print('skor=',round(linear_regressor.score(x,y),5))
print('ort hata=',toplam)

"""

"""
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=52)

from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(x_train,y_train)

y_predictions = linear_regressor.predict(x_test)


hata=y_predictions-y_test

hata=hata.reset_index()
hata=hata.fiyat
m=len(hata)
numb=0
for i in range(m):
    
    numb +=abs(hata[i])

toplam=round(numb/m,1)



m=len(hata)
numb=0
for i in range(m):
    if abs(hata[i])>50000:
     numb=numb+1 
    
    
print('skor=',round(linear_regressor.score(x_test,y_test),5))
print('ort hata=',toplam)    






from sklearn.metrics import mean_absolute_error , mean_squared_log_error,mean_squared_error
import math
print(mean_absolute_error(y_test,y_predictions))

print(median_absolute_error(y_test,y_predictions))
print(math.sqrt(mean_squared(y_test,y_predictions)))
"""




marka=dataframe.iloc[:, 0].values
yil=dataframe.iloc[:, 4].values
km=dataframe.iloc[:, 5].values
model=dataframe.iloc[:, 1].values
yakit=dataframe.iloc[:, 2].values
vites=dataframe.iloc[:, 3].values
"""
"""
"""
plt.scatter(marka,y, color="black")#marka gore

plt.scatter(marka, linear_regressor.predict(x),color="blue")


plt.scatter(yil,y, color="black")#yil gore

plt.scatter(yil, linear_regressor.predict(x),color="blue")
"""

plt.scatter(km,y, color="black")#km gore

plt.scatter(km, linear_regressor.predict(x),color="blue")

"""

plt.scatter(model,y, color="black")#yil gore

plt.scatter(model, linear_regressor.predict(x),color="blue")


plt.scatter(yakit,y, color="black")#yil gore

plt.scatter(yakit, linear_regressor.predict(x),color="blue")


plt.scatter(vites,y, color="black")#yil gore

plt.scatter(vites, linear_regressor.predict(x),color="blue")




"""


#Fitting model with trainig data


# Saving model to disk
pickle.dump(linear_regressor, open('model2.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
#print(model.predict([[2, 9, 6]]))