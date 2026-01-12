import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

def train_decision_tree(X_train, y_train, X_val, y_val):
    print("\n--- Training Decision Tree ---")
    # Grid Search for best params
    param_grid = {'max_depth': np.arange(1, 16, 1),
                  'criterion': ['entropy', 'gini'],
                  'max_features': ['sqrt', 'log2', None]}
    
    grid = GridSearchCV(DecisionTreeClassifier(random_state=123),
                        param_grid, scoring='roc_auc', cv=5)
    
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    
    print("Best Params:", grid.best_params_)
    print(f"Train AUC: {grid.best_score_:.4f}")
    print(f"Test AUC: {roc_auc_score(y_val, best_clf.predict(X_val)):.4f}")
    
    return best_clf

def train_mlp_sklearn(X_train, y_train, X_val, y_val):
    print("\n--- Training MLP (Sklearn) ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # Optimized params from your report
    model = MLPClassifier(random_state=1,
                          hidden_layer_sizes=(45, 45),
                          max_iter=200,
                          activation='logistic',
                          learning_rate_init=0.001,
                          alpha=0.01)
    
    model.fit(X_train_s, y_train)
    
    train_auc = roc_auc_score(y_train, model.predict(X_train_s))
    test_auc = roc_auc_score(y_val, model.predict(X_val_s))
    
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Confusion Matrix
    y_pred = model.predict(X_val_s)
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    
    return model, scaler

def train_svm(X_train, y_train, X_val, y_val):
    print("\n--- Training SVM ---")
    # Linear SVM optimized
    model = SVC(probability=True, kernel='linear', C=1.01)
    model.fit(X_train, y_train)
    
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    return model
