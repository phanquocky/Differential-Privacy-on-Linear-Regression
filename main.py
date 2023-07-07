import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# read file and visualize
df = pd.read_csv('lr.csv')
X = df['X']
y = df['Y']
X = X.values
y = y.values
X = X / np.max(X)
y = y / np.max(y)
X_DP = X
y_DP = y
plt.plot(X, y, '.')

# gradient descent
n = len(X)
ones = np.ones((len(X),1))
w_init = np.array([[1],[1]])
X = X.reshape(1, len(X))
y = y.reshape(len(y), 1)
X = np.concatenate([X.T, ones], axis= 1)
def gradient(X, w, y):   
    return X.T.dot((X.dot(w) - y)) / n

def gradient_descent(X, w, y, learning_reate = 0.001, iter = 1000000):
    for i in range(iter):
        w = w - learning_reate*gradient(X, w, y)
    return w

w = gradient_descent(X, w_init, y)
x = np.linspace(0, 1, 1000)
y = w[0][0] * x + w[1][0]
plt.plot(x,y)


# differential privacy
a0 = np.sum(y_DP ** 2)
a1 = -2 * np.sum(y_DP * X_DP)
a2 = np.sum(X_DP ** 2)
delta = 8
epsilon = 5

a0 = a0 + np.random.laplace(scale = delta / epsilon, loc = 0)
a1 = a1 + np.random.laplace(scale= delta / epsilon, loc = 0)
a2 = a2 + np.random.laplace(scale = delta / epsilon, loc = 0)

x0 = - a1 / (2 * a2) 
x = np.linspace(0, 1, 1000)
y = x0 * x 
plt.plot(x,y)
plt.show()

