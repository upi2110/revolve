# File: train/train_lstm.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from predictor.wheel_layout import EU_WHEEL
import os

LOOK_BACK = 10

def number_to_features(n):
    RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    BLACK = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
    return [
        n / 36.0,
        1 if n in RED else 0,
        1 if n in BLACK else 0,
        1 if n % 2 == 0 and n != 0 else 0,
        1 if n % 2 == 1 else 0,
        1 if n == 0 else 0,
        EU_WHEEL.index(n) / 36.0
    ]

with open("data/spins.txt") as f:
    numbers = [int(line.strip()) for line in f if line.strip().isdigit()]

X, y = [], []
for i in range(len(numbers) - LOOK_BACK):
    seq = numbers[i:i + LOOK_BACK]
    target = numbers[i + LOOK_BACK]
    features = [number_to_features(n) for n in seq]
    X.append(features)
    y.append(target)

X = np.array(X)
y = np.array(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], LOOK_BACK, -1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = Sequential([
    LSTM(64, input_shape=(LOOK_BACK, X_scaled.shape[-1])),
    Dropout(0.2),
    Dense(37, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(patience=5)])

os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
print("✅ LSTM model trained and saved.")