import numpy as np
import matplotlib.pyplot as plt

def LSE(pred, y):
    return (y - pred) ** 2
def gradient_LSE(betas, X_batch, y_batch):
    # Loss function: (y - pred)^2
    # Gradient w.r.t m and b: <2(y - pred) * d/dm(y - pred), 2(y - pred) * d/db(y - pred)>
    # = <2(y - (mx_i + b)) * -x_i, 2(y - (mx_i + b)) * 1>

    # y - (mx_i + b) = y_batch - (X_batch * betas) where X_batch is a nx2 matrix with first column all 1s and second col the actual x.
    prediction_diff = (y_batch - (np.cross(X_batch, betas))) * 2
    pass

np.random.seed(42)
# Vector of inputs
X = 2 * np.random.rand(100, 1)
# Vector of outputs
y = 4 + 3 * X + np.random.randn(100, 1)
# We'd like to find m and b that best fit this data.

def sgd(X, y, learning_rate, epochs, batch_size):
    X_size = len(X)
    X_bias = np.c_[np.ones((X_size, 1)), X]
    # [[m], [b]]
    beta = np.random.randn(2, 1)

    cost_history = []
    rng = np.random.default_rng()

    for epoch in range(epochs):
        # Move the rows only, keep the column the same.
        indices = rng.permutation(X_size)
        X_shuffle = X_bias[indices]
        y_shuffle = y[indices]

        # Instead of "true" SGD where we only calc for the loss function of one data point,
        # we calc a few data points to make convergence smoother.

        X_batch = X_shuffle[0:batch_size]
        y_batch = y_shuffle[0:batch_size]

        beta -= learning_rate * gradient_LSE(beta, 1, y_batch)



if __name__ == "__main__":
    sgd(X, y, 0.01, 1, 2)

