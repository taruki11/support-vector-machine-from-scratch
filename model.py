"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np
def standardize_features(x):
    x = np.asarray(x, dtype=float)

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)  # NumPy 默认计算总体标准差

    # 常数列不除以 0；减去均值后会变成 0
    safe_std = np.where(std == 0, 1.0, std)

    return (x - mean) / safe_std

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    return {
        "w": np.zeros(n_features, dtype=float),
        "b": 0.0,
    }

# Step 3 - compute_scores
import numpy as np
def compute_scores(x, params):
    x = np.asarray(x, dtype=float)
    w = np.asarray(params["w"], dtype=float)
    b = float(params["b"])

    return x @ w + b

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= 0, 1, -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    return float(max(0.0, 1.0 - y * score))

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(params["w"], dtype=float)

    scores = compute_scores(x, params)
    losses = np.maximum(0.0, 1.0 - y * scores)

    regularization = reg_lambda * np.dot(w, w)
    return float(np.mean(losses) + regularization)

# Step 7 - compute_gradients (not yet solved)
# TODO: implement

# Step 8 - apply_update (not yet solved)
# TODO: implement

# Step 9 - train_svm (not yet solved)
# TODO: implement

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

