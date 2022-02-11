#!/usr/bin/env python
# coding: utf-8

# In[17]:


import matplotlib as matplotlib

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=20,10

import tensorflow as tf  

from tensorflow import keras

from datetime import date, datetime

from keras.models import Sequential

import holidays

from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler


# In[18]:


def parse(x):
	return datetime.strptime(x, "%m/%d/%Y")
#Read csv and set parser to data 
df = pd.read_csv('D:\\source\\DataScience\\Cap_Plan_Hackathon_data.csv', index_col=0, date_parser=parse)
data_size=len(df)
train_size = data_size-90
print(data_size)
# manually specify column names
df.columns = ['No_calls_Offered']
df.index.name = 'Date'

df = df.astype({"No_calls_Offered": int})
df.dtypes


# In[19]:


#find out a date is a holiday or weekend
us_holidays = holidays.US()
def is_holiday_weekend(date):
    in_dt = date.strftime("%d-%m-%Y")
    is_holi = 0;
    if(in_dt in us_holidays or df["WeekDay"][i]==6 or df["WeekDay"][i]==5):
        is_holi = 1
    else:
        is_holi = 0
    return is_holi
        


# In[20]:


#findout the weekdays to identify weekends as holiday
df["WeekDay"]=df.index.weekday.astype('int')

is_holiday = []
for i in df.index:
    is_holiday.append(is_holiday_weekend(i))
    df["Day"] = i.day
    df["Month"] = i.month
    df["Year"] = i.year
    
df["Holiday"]=is_holiday
df = df.astype({"Holiday": int})
print(df["Holiday"])


# In[21]:


#plt.plot(df["Close"],label='Close Price history')


# In[22]:


df = df.sort_index(ascending=True,axis=0)


# In[23]:


#Identify the season

Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

def get_season(now):
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)
indx = 0
season = []
for i in df.index:
    season.append(get_season(i))
    
df["season"] = season


# In[24]:


print(df.head())


# In[25]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 


# In[ ]:





# In[26]:


values = df.values
# integer encode direction
encoder = LabelEncoder()
values[:,6] = encoder.fit_transform(values[:,6])
# ensure all data is int
values = values.astype('int')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
print(reframed.head())
 
# split into train and test sets
values = reframed.values
n_train_hours = train_size
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))

#from tensorflow.keras.optimizers import RMSprop
model.compile(loss='mae', optimizer='adam')
# fit network
#print(train_X)
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
#print(test_X)
# make a prediction
yhat = model.predict(test_X)
#yhat = np.argmax(yhat)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
#print(test_X[:, 1:])
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[27]:


print(inv_y)
train_data=df[:train_size]
valid_data=df[train_size:data_size-1]
print(inv_yhat)
valid_data['Predictions']=inv_yhat
plt.plot(train_data["No_calls_Offered"])
plt.plot(valid_data[['No_calls_Offered',"Predictions"]])


# In[ ]:





# In[ ]:





# In[ ]:




