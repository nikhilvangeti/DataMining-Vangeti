import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('mortgage_rates.csv') 
a = dataset.iloc[:,1:2].values
b = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 1/4, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(a_train, b_train)
b_pred = regressor.predict(a_test)

#Training Results
plt.scatter(a_train, b_train, color = 'red' )
plt.plot(a_train, regressor.predict(a_train), color = 'blue')
plt.title('interest_rates vs Medain_home_price (Training set)')
plt.xlabel('interest rates')
plt.ylabel('median home price')
plt.show()

#Test Results
plt.scatter(a_test, b_test, color = 'red' )
plt.plot(a_train, regressor.predict(a_train), color = 'blue')
plt.title('interest_rates vs Medain_home_price (Test set)')
plt.xlabel('interest rates')
plt.ylabel('median home price')
plt.show()