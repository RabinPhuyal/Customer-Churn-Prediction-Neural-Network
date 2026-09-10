# Customer Churn Prediction using PyTorch

A **neural network-based customer churn prediction project** built with Python and PyTorch. This project explores the complete workflow of a binary classification problem, from data preprocessing and exploratory data analysis to model training and evaluation.

> This project was built as part of my learning journey in **Deep Learning with PyTorch**, with a focus on understanding neural-network classification and applying it to a real-world dataset.

---

## 📌 Project Overview

Customer churn refers to customers discontinuing their service.

For subscription-based businesses, identifying customers who are likely to churn can help organizations take proactive steps to improve customer retention.

In this project, I built a **PyTorch neural network** to predict whether a customer is likely to churn based on customer and service-related information.

### Problem Statement

Given information about a customer, can a neural network predict whether the customer will:

* **Stay** with the service
* **Churn** from the service

This is treated as a **binary classification problem**.

---

## 🎯 Project Objectives

The main objectives of this project were to:

* Explore and understand the customer dataset
* Perform exploratory data analysis (EDA)
* Preprocess numerical and categorical features
* Prepare data for neural-network training
* Build a binary classification neural network using PyTorch
* Implement a training loop
* Evaluate model performance using multiple metrics
* Analyze model predictions using a confusion matrix
* Understand the practical challenges of customer churn prediction

---

## 🔍 Exploratory Data Analysis

Before building the model, I explored the dataset to understand patterns and relationships between customer characteristics and churn.

Some of the analysis included:

* Distribution of the churn target
* Customer churn across different internet service types
* Relationships between customer characteristics and churn
* Analysis of numerical features such as tenure and charges

Example visualization:

**Customer Churn by Internet Service**

The analysis helped identify differences in churn behavior across customer groups and provided context for the modeling stage.

---

## ⚙️ Data Preprocessing

The dataset required preprocessing before it could be used by the neural network.

The preprocessing workflow included:

* Handling missing or inconsistent values
* Converting numerical features to appropriate data types
* Encoding categorical variables
* Scaling numerical features
* Separating features (`X`) and target (`y`)
* Splitting the dataset into training and testing sets
* Converting the processed data into PyTorch tensors

---

## 🧠 Neural Network

The model was implemented using **PyTorch**.

The neural network performs binary classification and produces a prediction representing the likelihood of customer churn.

The overall workflow is:

```text
Customer Data
      ↓
Data Preprocessing
      ↓
PyTorch Tensors
      ↓
Neural Network
      ↓
Logits
      ↓
Sigmoid
      ↓
Churn Probability
      ↓
Churn / No Churn
```

### Key PyTorch concepts used

* `torch.Tensor`
* `nn.Module`
* `nn.Linear`
* Activation functions
* `BCEWithLogitsLoss`
* Optimizers
* Forward propagation
* Backpropagation
* Training loops
* Model evaluation

---

## 🏋️ Model Training

The neural network was trained using a custom PyTorch training loop.

The training process follows:

```text
Forward Pass
     ↓
Calculate Loss
     ↓
Zero Gradients
     ↓
Backpropagation
     ↓
Update Model Parameters
```

The model was trained over multiple epochs while monitoring its performance on the training and test data.

---

## 📊 Model Evaluation

Because this is a binary classification problem, I evaluated the model using multiple metrics rather than relying only on accuracy.

### Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **79.70%** |
| Precision | **63.92%** |
| Recall    | **54.01%** |
| F1 Score  | **58.55%** |

These results represent the current version of the model and may be improved through further experimentation and hyperparameter tuning.

---

## 🔲 Confusion Matrix

The confusion matrix provides a more detailed view of the model's predictions.

Current results:

|                     | Predicted No Churn | Predicted Churn |
| ------------------- | -----------------: | --------------: |
| **Actual No Churn** |                921 |             114 |
| **Actual Churn**    |                172 |             202 |

From the results:

* **True Negatives:** 921
* **True Positives:** 202
* **False Positives:** 114
* **False Negatives:** 172

### Why this matters

In a churn prediction problem, **false negatives are particularly important**.

A false negative means:

```text
Actual: Churn
Predicted: No Churn
```

This means the model failed to identify a customer who actually churned.

Therefore, future improvements should focus not only on accuracy, but also on improving **recall and F1 score**.

---

## 💡 Key Learning Outcomes

This project helped me gain practical experience with:

* Exploratory Data Analysis
* Data preprocessing
* Feature encoding
* Feature scaling
* Binary classification
* PyTorch tensors
* Neural network architecture
* Activation functions
* Loss functions
* Optimizers
* Backpropagation
* Custom training loops
* Model evaluation
* Confusion matrix analysis
* Precision, recall, and F1 score

More importantly, the project helped me understand how the concepts learned while studying PyTorch can be applied to an actual machine-learning problem.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **PyTorch**
* **Google Colab**
* **GitHub**

---

## 📂 Repository Structure

```text
Customer-Churn-Prediction-Neural-Network/
│
├── Customer_Churn_Prediction_Neural_Network.ipynb
└── README.md
```

The Jupyter/Colab notebook contains the complete workflow, including data exploration, preprocessing, model development, training, and evaluation.

---

## 🚧 Future Improvements

This project is currently focused on learning and applying neural-network classification with PyTorch.

Possible future improvements include:

* Experimenting with different neural-network architectures
* Hyperparameter tuning
* Improving recall and F1 score
* Comparing the neural network with traditional machine-learning models
* Exploring class imbalance techniques
* Performing additional feature analysis
* Saving and loading the trained model
* Deploying the model as an API

---

## 📚 Learning Context

This project is part of my ongoing journey in **Machine Learning, Deep Learning, and Artificial Intelligence**.

After completing neural-network regression and classification concepts in PyTorch, I applied those concepts to this customer churn prediction problem.

The next stage of my learning journey will focus on **Computer Vision and Convolutional Neural Networks (CNNs)**.

---

## 👨‍💻 Author

**Rabin Phuyal**

Data Science | Machine Learning | Artificial Intelligence

---

⭐ If you find this project useful or have suggestions for improvement, feel free to explore the notebook and share your feedback.
