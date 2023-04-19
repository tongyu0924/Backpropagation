import numpy as np
import matplotlib.pyplot as plt

train_x_flatten = np.random.randn(12288, 209)
train_x = train_x_flatten / 255
train_y = np.random.randn(1, 209)

def initialize_parameters(n_x, n_h, n_y):
  np.random.seed(1)
  W1 = np.random.randn(n_h, n_x) * 0.01
  b1 = np.zeros([n_h, 1])
  W2 = np.random.randn(n_y, n_h) * 0.01
  b2 = np.zeros([n_y, 1])

  parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

  return parameters

def forwardpass(X, parameters):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]

  Z1 = np.dot(W1, X) + b1
  A1 = np.maximum(0, Z1)
  Z2 = np.dot(W2, A1) + b2
  A2 = 1 / (1 + np.exp(-Z2))

  cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}

  return A2, cache




def compute_cost(A2, Y, parameters):
  m = Y.shape[1]

  cost = (1./m) * (-np.dot(Y, np.log(A2).T) - np.dot(1-Y, np.log(1 - A2).T))
  cost = np.squeeze(cost)

  return cost


def backwardpass(parameters, cache, X, Y):
  m = X.shape[1]

  W1 = parameters["W1"]
  W2 = parameters["W2"]
  A1 = cache["A1"]
  A2 = cache["A2"]
  Z1 = cache["Z1"]
  Z2 = cache["Z2"]

  dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

  temp_s = 1 / (1 + np.exp(-Z2))
  dZ2 = dA2 * temp_s * (1 - temp_s)

  dW2 = 1/m * np.dot(dZ2, A1.T)
  db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
  dA1 = np.dot(W2.T, dZ2)

  dZ1 = np.array(dA1, copy=True) 
  dZ1[Z1 <= 0] = 0  
    
  dW1 = 1/m * np.dot(dZ1, X.T)
  db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
  grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
  return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 5000, learning_rate=0.08, print_cost = False):
    costs = []
    
    np.random.seed(1)
    n_x = X.shape[0]
    n_y = Y.shape[0]
  
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):
        A2, cache = forwardpass(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backwardpass(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 500 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration {}: {}".format(i, cost))
   
    print("Cost after iteration {}: {}".format(i, cost))
    costs.append(cost)
    
    plt.figure(num=1, figsize=(8,5))
    plt.semilogy(costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Learning Rate = " + str(learning_rate))
    plt.show()
 
    return parameters

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1,m))
    
  
    probas, caches = forwardpass(X, parameters)
 
    
   
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


parameters = nn_model(train_x, train_y, 12, num_iterations=2500, learning_rate=0.007, print_cost=True)
