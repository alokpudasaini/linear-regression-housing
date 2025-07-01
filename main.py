import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data.csv")

X = df[['Size', 'Bedrooms']].values
y = df['Price'].values                 
m = len(y)                              

# Normalize features
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma


X_padded = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

def compute_cost(X, y, theta):
    m = len(y)
    h = X @ theta
    error = h - y
    return (1/(2*m)) * np.sum(error ** 2)

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        h = X @ theta
        theta = theta - (alpha/m) * (X.T @ (h - y))
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

# Initialize
theta = np.zeros(X_padded.shape[1])
alpha = 0.01
iterations = 400

# Run gradient descent
theta, J_history = gradient_descent(X_padded, y, theta, alpha, iterations)

print("Learned parameters (theta):", theta)

# Predict price of a 1650 sq-ft, 3 bedroom house
x_new = np.array([1650, 3])
x_norm = (x_new - mu) / sigma
x_padded = np.insert(x_norm, 0, 1)
predicted_price = x_padded @ theta

print(f"Predicted price: ${predicted_price:.2f}")

plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (Mean Squared Error)')
plt.title('Convergence of Gradient Descent')
plt.grid(True)
plt.show()

