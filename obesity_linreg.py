import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

#importing cleaned and merged dataset with median income and obesity rate by county
df_sample = pd.read_csv('mergedfiles.txt', sep='\t')
df_sample = df_sample[np.isfinite(df_sample['percent'])]
df_sample = df_sample[np.isfinite(df_sample['Median_Household_Income_2016'])]

X = df_sample[['Median_Household_Income_2016']]
Y = df_sample[['percent']]


#splitting data into test and training sets for linear regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.75, random_state=92) 

model = LinearRegression(fit_intercept=True)
model.fit(X_train, Y_train)

#making predictions using the test set 
y_predict = model.predict(X_test)

#assessing the fit of the linear regression model
coeff=model.coef_
print('Coefficients: \n', coeff)

print("Mean squared error: %.2f" % mean_squared_error(Y_test, y_predict))
r2=r2_score(Y_test, y_predict)
print('Variance score: %.2f' % r2_score(Y_test, y_predict))

#plot outputs
plt.plot(X,Y,'.')
plt.plot(X_test,y_predict, color='blue', linewidth=3)
plt.xlabel('Median Household Income, $')
plt.ylabel('Obesity %')
plt.text(100000,40, 'Variance score: %.2f' % r2)
plt.show()






