
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import os 
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df =  pd.read_csv('stocks.csv')
print(df.shape)
print(df.head())

"""Cleaning the data"""

df_new = df.drop(columns=['Ticker'])

print(df_new.head())

df_new=df_new.head(100)
print(df_new.shape)

"""visualization"""

# 1. Ensure 'Date' is datetime and sorted
df_new['Date'] = pd.to_datetime(df_new['Date'])
df_new.sort_values('Date',inplace=True)

# 2. Plotting
fig, ax = plt.subplots(figsize=(20,6))
ax.plot(df_new['Date'], df_new['Close'], label='Close')
ax.plot(df_new['Date'], df_new['Open'], label='Open')

# 3. Clean up labels
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Stock Price Over Time')
ax.legend()

# # 4. Fix the x-axis overlap
fig.autofmt_xdate()

plt.show()

"""Outliers"""

plt.figure(figsize=(12, 6))
plt.plot(df_new['Date'],df_new['Volume'],label='Volume')

plt.show()

# Drop non-numeric columns
numeric_data = df_new.select_dtypes(include=["int64","float64"])

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
# plt.show()

dataset = df_new.filter(['Close']).values

training_data_len = int(np.ceil(len(dataset) * 0.95))

# 1. Split the raw data first
train_raw = dataset[:training_data_len]


# 2. Fit the scaler ONLY on the training data
scaler = MinMaxScaler(feature_range=(0,1))
training_data = scaler.fit_transform(train_raw)

# 3. Transform the test data using the training parameters
# Note: Use .transform(), NOT .fit_transform()

"""Sliding window and converting array into 3 d for LSTM"""

X_train = []
y_train = []

for i in range(60, len(training_data)):
  X_train.append(training_data[i-60:i,0])
  y_train.append(training_data[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

"""Model Building"""

model = keras.models.Sequential()

#adding the first layer
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1],1)))

#Layer 2
model.add(keras.layers.LSTM(units=64, return_sequences=False))

#Layer 3
model.add(keras.layers.Dense(units=128,activation='relu'))

#Layer 4
model.add(keras.layers.Dropout(0.2))

#output layer
model.add(keras.layers.Dense(units=1))

model.summary()
optimizers = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizers, loss='mae',metrics=[keras.metrics.RootMeanSquaredError()])

model.fit(X_train, y_train, batch_size=4, epochs=20)

"""preparing the testing data"""

scalar1= MinMaxScaler(feature_range=(0,1))
test_raw = dataset[training_data_len-60:]
test_data = scalar1.fit_transform(test_raw)
X_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

"""Ploting"""

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# Plotting data
train = df_new[:training_data_len]
test =  df_new[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['Date'], train['Close'], label="Train (Actual)", color='blue')
plt.plot(test['Date'], test['Close'], label="Test (Actual)", color='orange')
plt.plot(test['Date'], test['Predictions'], label="Predictions", color='red')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()