# File: main.py

from predictor.predict_lstm import predict_single_sequence
from predictor.hybrid_zone_predictor import hybrid_zone_predict
from predictor.predict_xgb import predict_xgb

def main_menu():
    while True:
        print("\n📊 QUANTUM AI PREDICTOR MENU")
        print("1. Predict next number (LSTM)")
        print("2. Batch test from spins.txt (LSTM)")
        print("3. Predict next number (Smarter Hybrid Zone)")
        print("4. Predict using XGBoost")
        print("5. Exit")
        choice = input("Enter choice: ")

        if choice == '1':
            predict_single_sequence()  # LSTM prediction
        elif choice == '2':
            predict_batch_from_file("data/spins.txt")  # Batch testing for LSTM
        elif choice == '3':
            hybrid_zone_predict()  # Hybrid zone prediction (LSTM + XGB)
        elif choice == '4':
            predict_xgb()  # XGBoost prediction (standalone)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Try again.")

# Start the menu when the program runs
if __name__ == "__main__":
    main_menu()
