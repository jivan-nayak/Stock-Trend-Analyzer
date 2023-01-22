import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from keras.models import load_model
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler


yf.pdr_override()

symbol = 'AAPL'
# data_source='google'
start_date = '2012-01-01'
end_date = '2023-01-01'

st.title("Stock Trend Analyser")
user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = data.get_data_yahoo(user_input, start_date, end_date)

# Describing data
st.subheader("Data from 2012 to 2023")
st.write(df.describe())

# Visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("100 Day Moving Average MVA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader("200 Day Moving Average MVA")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

# Splitting data into training and testing

training_data = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
testing_data = pd.DataFrame(df['Close'][int(len(df) * 0.7): len(df)])

scaler = MinMaxScaler(feature_range=(0, 1))

training_data_array = scaler.fit_transform(training_data)

x_train = []
y_train = []

for i in range(100, training_data_array.shape[0]):
    x_train.append(training_data_array[i - 100: i])
    y_train.append(training_data_array[i, 0])

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

# Load Model
model = load_model('keras_model.h5')

past_100_days = training_data.tail(100)
final_df = past_100_days.append(testing_data, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.asarray(x_test).astype(np.float32), np.asarray(y_test).astype(np.float32)

y_predicted = model.predict(x_test)
factor = 1/scaler.scale_[0]


y_predicted = y_predicted * factor
y_test = y_test * factor

plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# plt.show()
st.write("scaler used is" + str(factor))
st.pyplot(plt)
