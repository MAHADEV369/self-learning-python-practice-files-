import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def LR(features, weights, bias):
    return(sigmoid(np.dot(features, weights) + bias))

features = np.array([1,2,3])
weights = np.array([0.5, 0.3, 0.1])
bias = -0.2
predicted_prob = LR(features, weights, bias)
print("predicted probability", predicted_prob)