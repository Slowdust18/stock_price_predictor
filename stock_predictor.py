# Importing required Libraries
import streamlit as st 
import numpy as np
import pandas as pd
from keras.models import load_model
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Setting Layout as wide
st.set_page_config(layout="wide")

# Title
st.title("Stock Price Predictor")

colinput, coltable = st.columns([1, 1])

with colinput:
    # Getting user input Google stocks code by default
    stock=st.text_input("Enter the stock ID", "GOOG")
    
    # Show url for Yahoo finance
    url = "https://finance.yahoo.com/"
    st.write("We use Yahoo Finance to train and test the model so use the below link to get stock codes")
    st.write("For stock codes go to this website [link](%s)" % url)

# Getting current date
end=datetime.now()

# Getting date 20 years back
start=datetime(end.year-20, end.month, end.day)

# Using try-catch block to end execution in case of am invalid stock code 
try:
    google_data = yf.download(stock, start, end)
    
    # Check if the data is empty
    if google_data.empty:
        st.error("The entered stock ID is invalid or has no data. Please try a different stock ID.")
        st.stop()  # Stop further execution

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Loading the created model
model=load_model("stock_prediction_model.keras")

with coltable:
    #Setting subheader
    st.subheader("Stock Data")
    st.write(google_data)


# Splitting the data in 7:3 format for training and testing 
# Since we need older data to train on and use the newer data to test on this is what is done
split_len=int(len(google_data)*0.7)
x_test=pd.DataFrame(google_data.Close[split_len:])
x_test.reset_index(drop=True, inplace=True)  # Flatten the index
x_test.columns = ['Close']


# Function for plotting graphs
def plot_graph(figsize, values, data, extra_data=0, extra_dataset=None):
    fig=plt.figure(figsize=figsize)
    plt.plot(values, "Orange")
    plt.plot(data.Close, "Green")
    if extra_data:
        plt.plot(extra_dataset)
    return fig

col1, col2 = st.columns(2)

# Original Close price and MA for 250 days
google_data['MA for 250 days'] = google_data.Close.rolling(250).mean()
with col1:
    st.subheader('Close Price & MA (250 days)')
    google_data['MA for 250 days']=google_data.Close.rolling(250).mean()
    st.pyplot(plot_graph((7, 4), google_data['MA for 250 days'], google_data))

# Original Close price and MA for 200 days
google_data['MA for 200 days'] = google_data.Close.rolling(200).mean()
with col2:  # Second column
    st.subheader('Close Price & MA (200 days)')
    google_data['MA for 200 days']=google_data.Close.rolling(200).mean()
    st.pyplot(plot_graph((7, 4), google_data['MA for 200 days'], google_data))

col3, col4 = st.columns([1, 1])
# Original Close price and MA for 100 days
with col3:
    st.subheader('Original Close price and MA for 100 days')
    google_data['MA for 100 days']=google_data.Close.rolling(100).mean()
    st.pyplot(plot_graph((7,4), google_data['MA for 100 days'], google_data))

# Original Close price and MA for 100 days and MA for 250 days
with col4:
    st.subheader('Original Close price and MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((7,4), google_data['MA for 100 days'], google_data,1,google_data['MA for 250 days']))


# Scaling the Close column down for easier training
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(x_test[['Close']])

# Splitting the dataset
x_data=[]
y_data=[]

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

# Converting list to arrays
x_data, y_data=np.array(x_data), np.array(y_data)

# Predicting using the model
predictions=model.predict(x_data)

# Reversing the scaled down values to get relevant values
actual_prediction=scaler.inverse_transform(predictions)
actual_expected=scaler.inverse_transform(y_data)


# Data frame for Original vs Predicted values
plot_data=pd.DataFrame(
    {
        'Original Test Data':actual_expected.reshape(-1),
        'Predicted Values':actual_prediction.reshape(-1),
    },
    index=google_data.index[split_len+100:]
)

col5, col6 = st.columns([1, 1])

# Original vs Predicted values
with col5:
    st.subheader("Original Values vs Predicted Values")
    st.write(plot_data)

# Original Close price vs predicted close price
with col6:
    st.subheader("Original Close Price vs Predicted Close Price")
    fig=plt.figure(figsize=(10,4))
    plt.plot(pd.concat([google_data.Close[:split_len+100], plot_data], axis=0))
    plt.legend(["Data- not used", "Original Test Data", "Predicted Test Data"])
    st.pyplot(fig)

# The end