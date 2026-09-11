# Customer Churn Prediction Using PyTorch

A customer churn prediction project built with **PyTorch**, focused on understanding the complete workflow of a binary classification problem — from data preprocessing and exploratory data analysis to neural network training, evaluation, and improving recall.

## 📌 Project Overview

Customer churn prediction is a common machine learning problem where the goal is to identify customers who are likely to leave a service.

In this project, I built a neural network using PyTorch to predict whether a customer will churn.

Rather than focusing only on accuracy, I also investigated **precision, recall, F1 score, and the confusion matrix**, with particular attention to improving recall because missing an actual churner can be costly in a churn prediction scenario.

## 🎯 Objectives

* Understand and preprocess customer churn data
* Perform exploratory data analysis
* Encode categorical features
* Scale numerical features
* Build a binary classification neural network using PyTorch
* Implement a custom training loop
* Evaluate the model using multiple classification metrics
* Analyze false positives and false negatives
* Investigate the precision-recall trade-off
* Improve churn detection using:

  * Decision threshold tuning
  * Class-weighted `BCEWithLogitsLoss`

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* PyTorch
* Google Colab

## 🔎 Exploratory Data Analysis

The project includes exploratory analysis to understand relationships between customer characteristics and churn.

Some of the analysis includes:

* Churn distribution
* Categorical feature analysis
* Internet service vs. churn
* Visualization of churn patterns

## 🧠 Neural Network Architecture

The model uses a simple feed-forward neural network:

```text
Input Features
      ↓
Linear Layer → 32 neurons
      ↓
ReLU
      ↓
Linear Layer → 16 neurons
      ↓
ReLU
      ↓
Linear Layer → 1 neuron
      ↓
Logit
```

The model outputs a single logit for binary classification.

`BCEWithLogitsLoss` is used because it combines the sigmoid operation and binary cross-entropy loss in a numerically stable way.

## ⚙️ Training

The model was trained using a custom PyTorch training loop involving:

* Forward pass
* Loss calculation
* Gradient clearing
* Backpropagation
* Optimizer step
* Training and evaluation modes
* Inference using `torch.inference_mode()`

The optimizer used was Adam with a learning rate of `0.001`.

## 📊 Baseline Results

The initial model produced:

| Metric    | Baseline |
| --------- | -------: |
| Accuracy  |   79.70% |
| Precision |   63.92% |
| Recall    |   54.01% |
| F1 Score  |   58.55% |

The most important issue was the relatively low recall.

The confusion matrix showed:

* True Positives: 202
* True Negatives: 921
* False Positives: 114
* False Negatives: 172

This meant the model was missing a significant number of actual churners.

## 🔧 Improving Recall

### 1. Decision Threshold Tuning

The initial model used a threshold of `0.50` to convert predicted probabilities into class labels.

I experimented with lower thresholds to investigate the precision-recall trade-off.

The best result from the original model was:

| Threshold |  Precision |     Recall |         F1 |
| --------: | ---------: | ---------: | ---------: |
|      0.50 |     65.90% |     45.99% |     54.17% |
|      0.40 |     60.16% |     60.16% |     60.16% |
|  **0.30** | **55.31%** | **72.46%** | **62.73%** |
|      0.25 |     52.29% |     76.20% |     62.02% |

This demonstrated that lowering the classification threshold significantly increased recall, although precision decreased.

### 2. Class-Weighted Loss

To make the model pay more attention to the churn class, I experimented with a class-weighted `BCEWithLogitsLoss`.

The weighted model produced:

| Metric    | Class-Weighted Model |
| --------- | -------------------: |
| Accuracy  |               75.51% |
| Precision |               52.73% |
| Recall    |               74.87% |
| F1 Score  |               61.88% |

I then combined class weighting with threshold tuning.

At a threshold of `0.40`:

| Metric    | Final Selected Configuration |
| --------- | ---------------------------: |
| Precision |                   **50.08%** |
| Recall    |                   **82.89%** |
| F1 Score  |                   **62.44%** |

This significantly improved recall compared with the original model.

## 📈 Key Learning

One of the main lessons from this project was that **accuracy alone does not tell the complete story**, especially for classification problems where one class may be more important than the other.

The baseline model achieved approximately 80% accuracy but had relatively low recall.

Through threshold tuning and class-weighted loss, I was able to increase recall substantially.

This helped me understand the trade-off between:

* True Positives
* False Positives
* False Negatives
* Precision
* Recall
* F1 Score
* Classification threshold

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Comparing with Logistic Regression and Random Forest
* ROC-AUC and Precision-Recall curves
* More detailed false-negative analysis
* Feature importance analysis
* Model deployment using FastAPI
* Containerization with Docker

## 📚 What I Learned

Through this project, I practiced:

* Data preprocessing
* Exploratory data analysis
* Binary classification
* Neural network architecture
* PyTorch tensors and DataLoaders
* Forward propagation
* Backpropagation
* Loss functions
* Optimizers
* Custom training loops
* Classification metrics
* Confusion matrix analysis
* Decision threshold tuning
* Class-weighted loss
* Precision-recall trade-offs

## 🔗 Project

GitHub Repository:

https://github.com/RabinPhuyal/Customer-Churn-Prediction-Neural-Network
