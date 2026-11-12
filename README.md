# Logistic Regression with Optimization Algorithms

## Project Overview

This project implements **Logistic Regression for Binary Classification** using five different optimization algorithms.

The main goal is to build Logistic Regression from scratch using **NumPy**, train the model using different optimizers, and compare their performance using standard classification metrics.

### Optimization Algorithms

1. GD with Momentum
2. SGD with Momentum
3. Mini-Batch GD with Momentum
4. RMSProp with Momentum
5. Adam

---

## Dataset

### Breast Cancer Wisconsin Diagnostic Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset** available through `scikit-learn`.

- Number of samples: **569**
- Number of input features: **30**
- Number of target classes: **2**
- Target classes:
  - `0` → Malignant
  - `1` → Benign
- Training samples: **455**
- Testing samples: **114**

The dataset is also included in this repository as:

`breast_cancer_wisconsin.csv`

---

## Data Preprocessing

The following preprocessing steps were applied:

### 1. Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

A `random_state` of 42 was used to ensure reproducibility.

### 2. Feature Standardization

The input features were standardized using `StandardScaler`.

The scaler was fitted only on the training data and then applied to the testing data.

### 3. Sigmoid Function

Logistic Regression uses the sigmoid function to convert the model output into a probability between 0 and 1.

### 4. Binary Cross-Entropy Loss

Binary Cross-Entropy was used as the loss function during training.

---

## Model Implementation

The Logistic Regression model was implemented using **NumPy**.

The implementation includes:

- Forward propagation
- Sigmoid activation
- Gradient calculation
- Binary Cross-Entropy Loss
- Weight updates
- Bias updates
- Training loss tracking
- Binary classification using a threshold of 0.5

---

## Optimization Algorithms

### GD with Momentum

Gradient Descent updates the model parameters using the gradients calculated from the entire training dataset.

Momentum uses the previous update direction to help accelerate convergence and reduce oscillations.

### SGD with Momentum

Stochastic Gradient Descent updates the parameters using one training sample at a time.

Momentum is applied to stabilize the updates.

### Mini-Batch GD with Momentum

Mini-Batch Gradient Descent updates the model using small batches of training samples.

A batch size of **32** was used.

Momentum is applied to the parameter updates.

### RMSProp with Momentum

RMSProp adapts the learning rate for each parameter using a moving average of squared gradients.

Momentum was additionally applied to the updates.

### Adam

Adam combines ideas from momentum and adaptive learning rates by maintaining first and second moment estimates of the gradients.

The following parameters were used:

- `beta1 = 0.9`
- `beta2 = 0.999`
- `epsilon = 1e-8`

---

## Evaluation Metrics

The following metrics were calculated for every optimizer:

- Accuracy
- Precision
- Recall
- F1 Score
- Test Loss
- Training Loss

---

## Results

| Optimizer | Training Loss | Test Loss | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|
| GD with Momentum | 0.0612 | 0.0567 | 0.9825 | 0.9859 | 0.9859 | 0.9859 |
| SGD with Momentum | 0.0352 | 0.0930 | 0.9649 | 0.9855 | 0.9577 | 0.9714 |
| Mini-Batch GD with Momentum | 0.0569 | **0.0553** | **0.9912** | 0.9861 | **1.0000** | **0.9930** |
| RMSProp with Momentum | 0.0384 | 0.0745 | 0.9825 | 0.9859 | 0.9859 | 0.9859 |
| Adam | 0.0941 | 0.0807 | **0.9912** | 0.9861 | **1.0000** | **0.9930** |

---

## Performance Comparison

The results show that:

- **Mini-Batch GD with Momentum** achieved the lowest Test Loss: **0.0553**.
- **Mini-Batch GD with Momentum** and **Adam** achieved the highest Accuracy: **0.9912**.
- Both achieved a Recall of **1.0000**.
- Both achieved the highest F1 Score: **0.9930**.
- SGD with Momentum achieved the lowest Training Loss, but its test performance was lower than the other optimizers.

Based on the overall test performance, **Mini-Batch GD with Momentum** was selected as the best optimizer because it achieved the lowest Test Loss while also achieving the highest Accuracy and F1 Score.

---

## Project Structure

```text
Logistic-Regression-Optimizers/
│
├── 01_Logistic_Regression_GD_Momentum.ipynb
├── 02_Logistic_Regression_SGD_Momentum.ipynb
├── 03_Logistic_Regression_MiniBatch_Momentum.ipynb
├── 04_Logistic_Regression_RMSProp_Momentum.ipynb
├── 05_Logistic_Regression_Adam.ipynb
├── 06_Optimizers_Comparison.ipynb
│
├── breast_cancer_wisconsin.csv
│
└── README.md
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- GitHub

---

## Conclusion

This project demonstrates the implementation and comparison of different optimization algorithms for Logistic Regression in a binary classification problem.

The comparison shows that optimizer performance should not be evaluated only by training loss. Test-set performance is also important when selecting the most suitable optimizer.

Among the five tested algorithms, **Mini-Batch GD with Momentum** provided the strongest overall performance on the test data.
