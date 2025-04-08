import numpy as np
import matplotlib.pyplot as plt

def LSE(pred, y):
    return (pred - y) ** 2
def gradient_LSE(betas, X_batch, y_batch):
    """Naively return the gradient of LSE loss function.
    Return: np.array[[change in y-intercept], [change in slope]]
    """
    # Loss function: (pred_i - y_i)^2
    # Gradient w.r.t b and m: <2(pred_i - y) * d/db(pred_i - y), 2(pred_i - y) * d/dm(pred_i - y)>
    # = <2(mx_i + b - y_i) * 1, 2(mx_i + b - y_i) * x_i>

    gradients = np.zeros((2, 1))
    b = betas[0][0]
    m = betas[1][0]
    for i in range(len(X_batch)):
        x_i = X_batch[i][1] # The X_bias is [[1, x_i], ...] and we want x_i
        y_i = y_batch[i][0]
        prediction_diff = 2 * (m * x_i + b - y_i)
        gradient = np.array([[prediction_diff], [prediction_diff * x_i]])
        gradients += gradient
    return gradients

def gradient_LSE_lnalg(betas, X_batch, y_batch):
    """Return the gradient of LSE loss function with style.
    Return: np.array[[change in y-intercept], [change in slope]]
    """
    # https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/13/lecture-13.pdf
    return 2 * X_batch.T.dot(X_batch.dot(betas) - y_batch)

np.random.seed(42)
# Vector of inputs
X = 2 * np.random.rand(100, 1)
# Vector of outputs
y = 4 + 3 * X + np.random.randn(100, 1)
# We'd like to find m and b that best fit this data.

def sgd(X, y, learning_rate, epochs, batch_size):
    X_size = len(X)
    X_bias = np.c_[np.ones((X_size, 1)), X]
    # [[b], [m]]
    betas = np.random.randn(2, 1)

    cost_history = []

    for epoch in range(epochs):
        # Move the rows only, keep the column the same.
        indices = np.random.permutation(X_size)
        X_shuffle = X_bias[indices]
        y_shuffle = y[indices]

        # For each epoch, we go through all points at least once.
        for i in range(0, X_size, batch_size):
            X_batch = X_shuffle[i:i + batch_size]
            y_batch = y_shuffle[i:i + batch_size]

            gradient = gradient_LSE(betas, X_batch, y_batch)
            betas -= learning_rate * gradient / batch_size

        # [1, x] * [[b], [m]] -> [b + x * m]
        predictions = X_bias @ betas
        cost = np.mean(LSE(predictions, y))
        cost_history.append(cost)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, cost: {cost}")
            print(f"Beta:\n{betas}")
            print("===================")
    
    print(f"Final: {betas}")

    return betas, cost_history


if __name__ == "__main__":
    sgd(X, y, 0.1, 1000, 10)

