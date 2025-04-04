import os
import json
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from predictor.wheel_layout import get_neighbors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

LOOK_BACK = 10
MODEL_LSTM_PATH = "models/lstm_model.h5"
MODEL_XGB_PATH = "models/xgb_model.json"
LOG_PATH = "logs/hybrid_log.json"

RED = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
BLACK = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}

EU_WHEEL = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34,
    6, 27, 13, 36, 11, 30, 8, 23, 10, 5,
    24, 16, 33, 1, 20, 14, 31, 9, 22, 18,
    29, 7, 28, 12, 35, 3, 26
]

def number_to_features(n):
    return [
        n / 36.0,
        1 if n in RED else 0,
        1 if n in BLACK else 0,
        1 if n % 2 == 0 and n != 0 else 0,
        1 if n % 2 == 1 else 0,
        1 if n == 0 else 0,
        EU_WHEEL.index(n) / 36.0  # new: normalized position on wheel
    ]

def preprocess_input(seq):
    features = [number_to_features(n) for n in seq]
    arr = np.array(features).reshape(1, LOOK_BACK, -1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr.reshape(-1, arr.shape[-1])).reshape(1, LOOK_BACK, -1)
    return scaled

def preprocess_input_flat(seq):
    return np.array([number_to_features(n) for n in seq]).flatten().reshape(1, -1)

def log_result(data):
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            json.dump([], f)
    with open(LOG_PATH, "r") as f:
        logs = json.load(f)
    logs.append(data)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)

def hybrid_zone_predict():
    print("\n🎯 SMART HYBRID PREDICTOR (Smarter Matching Logic Enabled)\n")
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

    lstm_model = load_model(MODEL_LSTM_PATH)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_XGB_PATH)

    while True:
        lstm_input = preprocess_input(last_10)
        xgb_input = preprocess_input_flat(last_10)

        lstm_probs = lstm_model.predict(lstm_input, verbose=0)[0]
        xgb_probs = xgb_model.predict_proba(xgb_input)[0]

        top_lstm_idx = int(np.argmax(lstm_probs))
        top_xgb_idx = int(np.argmax(xgb_probs))
        print(f"📊 Top LSTM: {top_lstm_idx} ({lstm_probs[top_lstm_idx]:.2%}), Top XGB: {top_xgb_idx} ({xgb_probs[top_xgb_idx]:.2%})")

        match_found = False
        for idx in np.argsort(lstm_probs)[::-1]:
            if lstm_probs[idx] >= 0.70:
                lstm_center = idx
                lstm_zone = get_neighbors(lstm_center, 1)
                for j in lstm_zone:
                    if xgb_probs[j] >= 0.70:
                        center = j
                        zone = get_neighbors(center, 4)
                        print(f"\n🎯 Matched Zone Found (Smarter Method)")
                        print(f"Center: {center} | LSTM: {lstm_probs[center]:.2%} | XGB: {xgb_probs[center]:.2%}")
                        print(f"Zone (9 numbers): {zone}")
                        match_found = True
                        break
                if match_found:
                    break

        if not match_found:
            print("🟡 No strong zone match found. Waiting...")
            try:
                next_val = input("➕ Enter next number to continue (or 'q' to quit): ").strip()
                if next_val.lower() == 'q': break
                last_10.pop(0)
                last_10.append(int(next_val))
            except:
                print("❌ Invalid input. Try again.")
            continue

        for attempt in range(1, 4):
            actual_input = input(f"[{attempt}/3] Enter actual number: ").strip()
            if actual_input.lower() == 'q': return
            try:
                actual = int(actual_input)
                result = "✅ HIT" if actual in zone else "❌ MISS"
                print(result)

                log_result({
                    "timestamp": datetime.now().isoformat(),
                    "center": center,
                    "zone": zone,
                    "actual": actual,
                    "attempt": attempt,
                    "result": result,
                    "lstm_conf": float(lstm_probs[center]),
                    "xgb_conf": float(xgb_probs[center])
                })

                if result == "✅ HIT":
                    break
            except:
                print("❌ Invalid input.")

        try:
            new_val = input("➕ Enter next number to continue (or 'q' to quit): ").strip()
            if new_val.lower() == 'q': break
            last_10.pop(0)
            last_10.append(int(new_val))
        except:
            print("❌ Invalid number.")
