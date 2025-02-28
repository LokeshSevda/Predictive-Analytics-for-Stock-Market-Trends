import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Fetch stock data
today = date.today()
end_date = today.strftime("%Y-%m-%d")
start_date = (today - timedelta(days=5000)).strftime("%Y-%m-%d")

data = yf.download('AAPL', start=start_date, end=end_date, progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)

# Visualizing stock data using Candlestick chart
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title="Apple Stock Price Analysis", xaxis_rangeslider_visible=False)
figure.show()

# Correlation analysis
correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))

# Prepare dataset for LSTM
x = data[["Open", "High", "Low", "Volume"]].to_numpy()
y = data["Close"].to_numpy().reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# LSTM Model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(xtrain, ytrain, batch_size=1, epochs=30)

# Test the model
features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
prediction = model.predict(features)
print("Predicted Stock Price:", prediction)
