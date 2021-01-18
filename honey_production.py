import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# This is a DataFrame.
df = pd.read_csv('honeyproduction.csv')

# groupby used to group columns
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# We need the years for the X axis
X = prod_per_year['year']

# one column with many rows
X = X.values.reshape(-1, 1)

# We need the production per year for the Y axis
y = prod_per_year['totalprod']

plt.scatter(X, y)

# Create LinearRegression object
regr = linear_model.LinearRegression()

regr.fit(X, y)

# We can look at the close and intercept.
#print(regr.coef_[0])
#print(regr.intercept_)

y_predict = regr.predict(X)

plt.plot(X, y_predict)

# We will now look at 2013-2050 which is the future.
X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)

# We predict using new X values
future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)
plt.show()
