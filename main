import argparse
import os
import pandas as pd
import pickle
from src.preprocessing import load_and_split_data, apply_preprocessing_train, apply_preprocessing_test
from src.models import train_decision_tree, train_mlp_sklearn, train_svm

def main(args):
    # 1. Load Data
    data_path = os.path.join('data', 'XY_train.pkl')
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}. Please place XY_train.pkl in 'data/' folder.")
        return

    print("Loading and splitting data...")
    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data(data_path)
    
    # 2. Preprocessing
    print("Preprocessing training data...")
    X_train_processed, feature_names = apply_preprocessing_train(X_train_raw.copy(), y_train)
    
    print("Preprocessing validation data...")
    X_test_processed = apply_preprocessing_test(X_test_raw.copy(), feature_names)
    
    # 3. Model Training
    model = None
    if args.model == 'dt':
        model = train_decision_tree(X_train_processed, y_train, X_test_processed, y_test)
    elif args.model == 'mlp':
        model, scaler = train_mlp_sklearn(X_train_processed, y_train, X_test_processed, y_test)
        # Save scaler if needed for inference later
        with open('data/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    elif args.model == 'svm':
        model = train_svm(X_train_processed, y_train, X_test_processed, y_test)
    
    # 4. Save Model
    if model:
        os.makedirs('data', exist_ok=True)
        save_path = f'data/model_{args.model}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline for Sentiment Analysis")
    parser.add_argument('--model', type=str, default='mlp', choices=['dt', 'mlp', 'svm'],
                        help="Choose model to train: 'dt' (Decision Tree), 'mlp' (Neural Net), 'svm' (SVM)")
    
    args = parser.parse_args()
    main(args)
