
import plotly.plotly as py
import plotly
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

#credentials for plotly API
plotly.tools.set_credentials_file(username='nbennett1', api_key='81baDAChISuvHTiv7rOQ')

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

#calculating deviation from linear regression model for each county
fullY_predict = model.predict(X)
Resid = Y-fullY_predict
df_sample['Resid']=Resid

#plotting the deviation from linear regression for each county
colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]
endpts = list(np.linspace(-16, 16, len(colorscale) - 1))
fips = df_sample['FIPS'].tolist()
values = df_sample['Resid'].tolist()

fig = ff.create_choropleth(
    fips=fips, values=values,
    binning_endpoints=endpts,
    colorscale=colorscale,
    show_state_data=False,
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title='Obesity Deviation from Linear Regression on Median Income, %',
    legend_title='marginal % obese'
)
py.iplot(fig, filename='choropleth_full_usa')
