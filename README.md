# Machine Learning Model Comparison for Sentiment Analysis 🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Research_Completed-green)

**Authors:** Jacob Maimon, Gal Sabaro  
**Institution:** Ben-Gurion University of the Negev  
**Course:** Machine Learning

---

## 📌 Overview
This project presents a comprehensive comparative analysis of supervised machine learning algorithms applied to **Sentiment Analysis** (classification of text as Positive/Negative).

We evaluated **Decision Trees**, **Artificial Neural Networks (ANN)**, and **Support Vector Machines (SVM)**, optimizing each model's hyperparameters to achieve maximum accuracy. The project also explores unsupervised clustering using K-Means and K-Medoids.

## ⚙️ Model Configuration (Best Hyperparameters)

Based on our research and Grid Search results, the following configurations yielded the best performance:

| Algorithm | Parameter | Optimized Value | Description |
| :--- | :--- | :--- | :--- |
| **Artificial Neural Network (MLP)** | `hidden_layer_sizes` | **(45, 45)** | Two hidden layers with 45 neurons each |
| | `activation` | `'logistic'` | Sigmoid activation function |
| | `alpha` | `0.01` | L2 Regularization penalty |
| | `learning_rate_init` | `0.001` | Initial step size for optimizer |
| **Support Vector Machine (SVM)** | `kernel` | `'linear'` | Effective for high-dimensional sparse text data |
| | `C` | `1.01` | Regularization parameter |
| **Decision Tree** | `criterion` | `'entropy'` | Metric for information gain |
| | `max_depth` | *Pruned* | Optimized to prevent overfitting |

## 🧪 Feature Engineering Pipeline

To process the raw text data, we implemented a robust pipeline:

1.  **Text Tokenization:** Splitting text and removing stop-words.
2.  **TF-IDF Vectorization:** Extracting up to **30,000** features based on term importance.
3.  **Feature Selection:** Using `SelectKBest` with ANOVA F-value (`f_classif`) to keep only the top **19-30** most discriminative features (reduced from 30k to improve runtime and reduce noise).

## 📊 Performance Results

The models were evaluated on a held-out test set (20% split). The Neural Network achieved the highest accuracy, demonstrating the effectiveness of the chosen architecture for this task.

| Model | Accuracy / AUC | Rank | Notes |
| :--- | :--- | :--- | :--- |
| **ANN (MLP)** | **~99.51%** | 🏆 **1st** | Best generalization using Dropout & Regularization |
| **SVM (Linear)** | High | 2nd | Very fast training, competitive results |
| **Decision Tree** | Moderate | 3rd | Good interpretability, but prone to overfitting |

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JacobMaimon13/ML-Classification-Comparison.git](https://github.com/JacobMaimon13/ML-Classification-Comparison.git)
    cd ML-Classification-Comparison
    ```

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis:**
    You can choose which model to train using the `--model` flag:

    ```bash
    # Run the best model (Neural Network)
    python main.py --model mlp

    # Run Decision Tree
    python main.py --model dt
    
    # Run SVM
    python main.py --model svm
    ```

## 📂 Project Structure

* `src/`: Contains the core logic (`preprocessing.py` for feature extraction, `models.py` for classifiers).
* `data/`: Datasets (`XY_train.pkl`, `X_test.pkl`) and saved models.
* `main.py`: The entry point for running the analysis.

---
*Based on the final project submitted for the Dept. of Industrial Engineering and Management, BGU.*
