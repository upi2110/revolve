# File: predictor/predict_xgb.py

import xgboost as xgb
import numpy as np
from predictor.wheel_layout import EU_WHEEL

MODEL_XGB_PATH = "models/xgb_model.json"

# Function to get neighbors of a number on the wheel
def get_neighbors(number, n=4):
    """Get 'n' neighbors on both sides of a given number from the European Roulette wheel"""
    index = EU_WHEEL.index(number)
    return [EU_WHEEL[(index + i) % len(EU_WHEEL)] for i in range(-n, n+1)]

# Function to preprocess the input data
def preprocess_input_flat(seq):
    # Returns the flattened list of features for the XGBoost model
    def number_to_features(n):
        RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
        BLACK = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
        return [
            n / 36.0,  # Normalized number
            1 if n in RED else 0,  # Red number
            1 if n in BLACK else 0,  # Black number
            1 if n % 2 == 0 and n != 0 else 0,  # Even
            1 if n % 2 == 1 else 0,  # Odd
            1 if n == 0 else 0,  # Zero
            EU_WHEEL.index(n) / 36.0  # Normalized wheel position
        ]

    return np.array([number_to_features(n) for n in seq]).flatten().reshape(1, -1)

# XGBoost Prediction function
def predict_xgb():
    print("\n🎯 Predicting using XGBoost model...\n")
    last_10 = []

    while len(last_10) < 10:
        try:
            val = input(f"[{len(last_10)+1}/10] ➜ Enter number: ").strip()
            if val.lower() == 'q': return
            n = int(val)
            assert 0 <= n <= 36
            last_10.append(n)
        except:
            print("❌ Invalid input. Enter a number 0-36.")

    # Load XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_XGB_PATH)

    while True:
        xgb_input = preprocess_input_flat(last_10)  # Now this works properly
        xgb_probs = xgb_model.predict_proba(xgb_input)[0]

        # Get top prediction
        top_xgb_idx = int(np.argmax(xgb_probs))
        confidence = xgb_probs[top_xgb_idx]

        # Show predicted number, confidence level, and its neighbors in the requested format
        predicted_number = top_xgb_idx
        neighbors = get_neighbors(predicted_number)

        # ANSI escape codes for bold and green color
        bold_green = '\033[1;32m'  # Green bold text
        reset = '\033[0m'  # Reset formatting

        # Output format with bold green color and confidence
        print(f"{bold_green}{predicted_number}{reset} [{len(neighbors)}-neighbours] | Confidence: {confidence:.2%}")

        try:
            next_val = input("➕ Enter next number to continue (or 'q' to quit): ").strip()
            if next_val.lower() == 'q': break
            last_10.pop(0)
            last_10.append(int(next_val))
        except:
            print("❌ Invalid input.")
