import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

LOOK_BACK = 10
MODEL_PATH = "models/lstm_model.h5"

RED = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

def number_to_features(n):
    return [
        n / 36.0,
        1 if n in RED else 0,
        1 if n in BLACK else 0,
        1 if n % 2 == 0 and n != 0 else 0,
        1 if n % 2 == 1 else 0,
        1 if n == 0 else 0
    ]

def preprocess_sequence(seq):
    features = [number_to_features(n) for n in seq]
    arr = np.array(features).reshape(1, LOOK_BACK, -1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr.reshape(-1, arr.shape[-1])).reshape(1, LOOK_BACK, -1)
    return scaled

def load_lstm_model():
    return load_model(MODEL_PATH)

def plot_top_predictions(probabilities):
    top_indices = np.argsort(probabilities)[-5:][::-1]
    top_probs = [probabilities[i] for i in top_indices]
    top_labels = [str(i) for i in top_indices]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(top_labels, top_probs)
    plt.title("Top 5 Predicted Numbers")
    plt.xlabel("Number")
    plt.ylabel("Probability")
    for bar, prob in zip(bars, top_probs):
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.25, yval + 0.01, f"{prob:.2%}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def predict_single_sequence():
    model = load_lstm_model()
    input_str = input("🔢 Enter last 10 numbers separated by space (latest last):\n")
    last_10 = [int(n) for n in input_str.strip().split()]
    assert len(last_10) == 10, "❌ Must enter exactly 10 numbers."

    X = preprocess_sequence(last_10)
    probs = model.predict(X)[0]
    predicted_number = np.argmax(probs)
    confidence = np.max(probs)

    print("\n📈 Top 5 Predictions:")
    plot_top_predictions(probs)

    if confidence >= 0.80:
        print(f"🎯 Prediction: {predicted_number} with {confidence:.2%} confidence ✅")
    else:
        print(f"🟡 Low confidence ({confidence:.2%}). Suggest waiting.")

def predict_batch_from_file(path):
    model = load_lstm_model()
    with open(path) as f:
        numbers = [int(line.strip()) for line in f if line.strip().isdigit()]

    total = 0
    confident_hits = 0
    for i in range(len(numbers) - LOOK_BACK - 1):
        seq = numbers[i:i+LOOK_BACK]
        actual = numbers[i+LOOK_BACK]
        X = preprocess_sequence(seq)
        probs = model.predict(X)[0]
        pred = np.argmax(probs)
        conf = np.max(probs)
        if conf >= 0.80:
            total += 1
            if pred == actual:
                confident_hits += 1
            print(f"[{i}] Predicted: {pred} | Actual: {actual} | Confidence: {conf:.2%}")

    print(f"\n📊 Confident Predictions: {total}")
    print(f"✅ Hits: {confident_hits}")
    if total > 0:
        print(f"🎯 Hit Rate: {confident_hits/total:.2%}")
    else:
        print("🟡 No confident predictions found.")
