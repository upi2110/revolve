# File: train/train_xgb.py

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from predictor.wheel_layout import EU_WHEEL
import os

LOOK_BACK = 10

RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
BLACK = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}

def number_to_features(n):
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
    X.append(np.array(features).flatten())
    y.append(target)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ XGBoost Accuracy: {accuracy * 100:.2f}%")

os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_model.json")
print("✅ XGBoost model trained and saved.")