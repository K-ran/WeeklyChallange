import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dataframe = pd.read_csv('challenge_dataset.txt',header=None)

x = dataframe[[0]]
y = dataframe[[1]]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x,y)

error = mean_squared_error(y,body_reg.predict(x))

plt.text(1,27,"Error(least Square): "+str(error))

plt.scatter(x,y)
plt.plot(x,body_reg.predict(x))
plt.show();