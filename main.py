import matplotlib.pyplot as plt
import numpy as np


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
        x_i = X_batch[i][1]  # The X_bias is [[1, x_i], ...] and we want x_i
        y_i = y_batch[i][0]
        prediction_diff = 2 * (m * x_i + b - y_i)
        gradient = np.array([[prediction_diff], [prediction_diff * x_i]])
        gradients += gradient
    return gradients


def gradient_LSE_lnalg(betas, X_batch, y_batch):
    """Return the gradient of LSE loss function with "sTyLe".
    Return: np.array[[change in y-intercept], [change in slope]]
    """
    # https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/13/lecture-13.pdf
    return 2 * X_batch.T @ (X_batch @ betas - y_batch)


np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# From one of the lecture example.
# X = np.array([[0], [1], [2]])
# y = np.array([[4], [3], [1]])

# We'd like to find m and b that best fit this data.


def sgd(X, y, *, lr, epochs, batch_size, momentum=0):
    X_size = len(X)
    X_bias = np.c_[np.ones((X_size, 1)), X]
    # [[b], [m]]
    betas = np.random.randn(2, 1)

    cost_history = []
    beta_history = []

    for epoch in range(epochs):
        # Move the rows only, keep the column the same.
        indices = np.random.permutation(X_size)
        X_shuffle = X_bias[indices]
        y_shuffle = y[indices]
        velocity = 0

        # For each epoch, we go through all points at least once.
        for i in range(0, X_size, batch_size):
            X_batch = X_shuffle[i : i + batch_size]
            y_batch = y_shuffle[i : i + batch_size]

            gradient = gradient_LSE_lnalg(betas, X_batch, y_batch)
            if i > 0:
                velocity = momentum * velocity + gradient
            else:
                velocity = gradient

            betas -= lr * velocity / batch_size

        # [1, x] * [[b], [m]] -> [b + x * m]
        predictions = X_bias @ betas
        cost = np.mean(LSE(predictions, y))

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, cost: {cost}")
            print(f"Beta:\n{betas}")
            print("===================")
            beta_history.append(betas)
            cost_history.append(cost)

    print(f"Final: b={betas[1][0]}, m={betas[0][0]}")
    predictions = X_bias @ betas
    cost = np.mean(LSE(predictions, y))
    print(f"Cost: {cost}")
    beta_history.append(betas)
    cost_history.append(cost)

    return beta_history, cost_history


def plot3d(beta_history, cost_history):
    spanning_radius = 1
    b_range = np.arange(4 - spanning_radius, 4 + spanning_radius, 0.05)
    m_range = np.arange(3 - spanning_radius, 3 + spanning_radius, 0.05)
    b_grid, m_grid = np.meshgrid(b_range, m_range)

    range_len = len(b_range)
    # Make [w0, w1] in (2, 14400) shape
    all_w0w1_values = np.hstack(
        [b_grid.flatten()[:, None], m_grid.flatten()[:, None]]
    ).T

    # Compute all losses, reshape back to grid format
    # Magic stuff, no idea what's happening here :)
    all_losses = (
        np.linalg.norm(
            y - (np.c_[np.ones((len(X), 1)), X] @ all_w0w1_values), axis=0, ord=2
        )
        ** 2
    ).reshape((range_len, range_len))

    print(((y - (np.c_[np.ones((len(X), 1)), X] @ all_w0w1_values))**2).shape)

    xs = []
    ys = []
    for betas in beta_history:
        b = betas[0][0]
        m = betas[1][0]
        xs.append(b)
        ys.append(m)

    def fmt(x):
        return f"{x:.1f}"

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(b_grid, m_grid, all_losses, alpha=0.5, cmap="RdBu")
    cs = ax.contour(b_grid, m_grid, all_losses, offset=0, alpha=1, cmap="RdBu")
    ax.clabel(cs, cs.levels, fmt=fmt, fontsize=10)
    
    ax.scatter(xs[:-1], ys[:-1], cost_history[:-1], c="green")
    ax.scatter(xs[-1], ys[-1], cost_history[-1], c="red")
    
    ax.set_xlabel("y-intercept")
    ax.set_ylabel("slope")
    ax.set_zlabel("L(w)")
    ax.set_xticks(np.arange(4 - spanning_radius, 4 + spanning_radius, 2))
    ax.set_yticks(np.arange(3 - spanning_radius, 3 + spanning_radius, 2))
    ax.set_zticks([])
    plt.show()


def plot_bestfit(X, y, beta_final):
    plt.scatter(X, y, color="blue", label="Data points")
    plt.plot(
        X,
        np.c_[np.ones((X.shape[0], 1)), X].dot(beta_final),
        color="red",
        label="SGD fit line",
    )
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression using Stochastic Gradient Descent")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    beta_history, cost_history = sgd(
        X,
        y,
        lr=0.1,
        epochs=1000,
        batch_size=10,
        momentum=0,
    )

    plot3d(beta_history, cost_history)
