import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from keras.models import load_model
import streamlit as st
start = '2017-10-20'
end = '2022-10-17'

st.title('Stock trend Prediction')


df = pdr.get_data_tiingo("AAPL",api_key = 'ba9ee346c09c22920c0e869b31a24db7f6cc1051')

# Describing the data
st.subheader('Data from 2017-2022')
st.write(df.describe())

# visualisation
st.subheader("Closing Price Visualisation")
df1 = df.reset_index()['close']
fig = plt.figure(figsize=(12,6))
plt.plot(df1)
st.pyplot(fig)

# scaling the data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

#splitting into training and testing data
training_size = int(len(df1)*0.65)
test_size = int(len(df1)*0.35)
train_data,test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]

def create_dataset(dataset,time_step = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX),np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and y=t+4
X_train,y_train = create_dataset(train_data,100)
X_test,y_test = create_dataset(test_data,100)

# reshape input to the [samples,time_steps,features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

#load my model

model = load_model('keras_model.h5')



# make predictions on the model
# Plotting
# shift train predictions for plotting
st.subheader('Plotting the next 30 days')
fig1 = plt.figure(figsize=(12,6))
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transformback to the original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

import tensorflow as tf

x_input = test_data[340:].reshape(1,-1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

from numpy import array
lst_output = []
n_steps=100
i=0
while i<30:
    if (len(temp_input)>100):
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape(1,n_steps,1)
        #print (x_input)
        yhat = model.predict(x_input,verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape(1,n_steps,1)
        yhat = model.predict(x_input,verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1

day_new = np.arange(1,101)
day_pred = np.arange(101,131)
plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


st.pyplot(fig1)

#%%
