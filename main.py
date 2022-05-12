import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import yfinance as yf

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


#  DATA to be read with data descriptions
stock = input("choose a stock")
data = yf.download(stock, "2010-01-01", "2022-01-01", auto_adjust=True)
P = data.head()
s = data.shape
i = data.info()
des = data.describe()

print(P)
print(s)
print(i)
print(des)

# visualize the data

data.Volume.plot(figsize=(10, 7), color='r')
plt.ylabel(f"{format(stock)} prices")
plt.title(f"{format(stock)} price series")
plt.show()

sn.distplot(data['Close'])
plt.show()


# TRAINING ADN TESTING

"""
This is not a multiline linear regression hence there is no y value
"""

x = data.drop(["Close", "Volume"], axis=1)
y = data["Close"]

"""
split data 80% training and 20 % teting
"""

x_train ,x_test , y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# MODELLING

lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)

# print(pred)

"""
using MSE AND MAE

y-test = ground value/truth
y_pred = model predicted value
"""

def calc(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    score = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)

    print(f"mse ,{mse}")
    print(f"mae, {mae}")
    print(f"r2 {score}")

calc(y_test,pred)

"""
implementing ridge and lasso regression
"""

la = Lasso().fit(x_train,y_train)
ri = Ridge().fit(x_train,y_train)

lp = la.predict(x_test)
rp = ri.predict(x_test)

calc(y_test,lp)
calc(y_test,rp)


# FINE TUNING
vr = SVR()


param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
gp = grid.fit(x_train, y_train)

print(gp)

svr = SVR(C=10, gamma=0.01, kernel='rbf')
svr.fit(x_train, y_train)
svr_pred = svr.predict(x_test)

print(svr_pred)
