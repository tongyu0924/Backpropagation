import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(12280, 209) / 255
Y = np.random.randn(1, 209)

def initialize_parameters(n_x, n_h, n_y):
  np.random.randn(1)
  W1 = np.random.randn(n_h, n_x) * 0.01
  b1 = np.zeros((n_h, 1))
  W2 = np.random.randn(n_y, n_h) * 0.01
  b2 = np.zeros((n_y, 1))

  parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
  
  return parameters

def forward(X, parameters):
  Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
  A1 = 1 / (1 + np.exp(-Z1))
  Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
  A2 = 1 / (1 + np.exp(-Z2))

  cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}

  return A2, cache

def compute_cost(A2, Y):
  m = Y.shape[1]

  cost = (-1/m) * (np.dot(Y, np.log(A2).T) + np.dot((1 - Y), np.log(1 - A2).T))
  cost = np.squeeze(cost)
  return cost

def backward(X, Y, cache):
  m = X.shape[1]

  Z1 = cache["Z1"]
  A1 = cache["A1"]
  Z2 = cache["Z2"]
  A2 = cache["A2"]

  grads = {}
  dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
  dZ2 = dA2 * (1 / 1 + np.exp(-Z2)) * (1 - (1 / 1 + np.exp(-Z2)))
  dW2 = 1/m * np.dot(dZ2, A1.T)
  db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
  dA1 = np.dot(dW2.T, dZ2)
  dZ1 = dA1 * (1 / 1 + np.exp(-Z1)) * (1 - (1 / 1 + np.exp(-Z1)))
  dW1 = 1/m * np.dot(dZ1, X.T)
  db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
  grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
  return grads

def update_parameters(parameters, grads, learning_rate):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]
    
  dW1 = grads["dW1"]
  db1 = grads["db1"]
  dW2 = grads["dW2"]
  db2 = grads["db2"]
    
  W1 = W1 - learning_rate * dW1
  b1 = b1 - learning_rate * db1
  W2 = W2 - learning_rate * dW2
  b2 = b2 - learning_rate * db2
    
  parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
  return parameters


np.random.seed(1)
costs = []
parameters = initialize_parameters(X.shape[0], 8, Y.shape[0])

for i in range(5):

  A2, caches = forward(X, parameters)
  cost = compute_cost(A2, Y)

  grads = backward(X, Y, caches)

  parameters = update_parameters(parameters, grads, 0.1)

  print(cost)
