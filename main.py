import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Data as vectors. X is input vector, Y is expected vector.
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# From one of the lecture example. Warning: the graph won't look nice because not enough (X, Y).
# X = np.array([[0], [1], [2]])
# Y = np.array([[4], [3], [1]])

# Configure SGD:
LEARNING_RATE = 0.1  # The smaller the slower it converges.
EPOCHS = 1000  # Around 5000 max is enough, no need to go crazy.
BATCH_SIZE = 10
MOMENTUM = 0.9  # Between 0 and 1; 0 is no momentum.

# Configure graph:
CENTER_ON = (4, 3)
CONTOUR_MIN = 0
CONTOUR_MAX = 400
CONTOUR_STEP = 25


# We'd like to find m and b that best fit this data.

# Don't change anything below.

# The regressor matrix. Convenient to have it here cuz we'll use it throughout.
X_bias = np.c_[np.ones((len(X), 1)), X]
betas = np.random.randn(2, 1)


def LSE(pred, y):
    return (pred - y) ** 2


def gradient_LSE(weights, X_batch, y_batch):
    """Naively return the gradient of LSE loss function.
    Return: np.array[[change in y-intercept], [change in slope]]
    """
    # Loss function: (pred_i - y_i)^2
    # Gradient w.r.t b and m: <2(pred_i - y) * d/db(pred_i - y), 2(pred_i - y) * d/dm(pred_i - y)>
    # = <2(mx_i + b - y_i) * 1, 2(mx_i + b - y_i) * x_i>

    gradients = np.zeros((2, 1))
    b = weights[0][0]
    m = weights[1][0]
    for i in range(len(X_batch)):
        x_i = X_batch[i][1]  # The X_bias is [[1, x_i], ...] and we want x_i
        y_i = y_batch[i][0]
        prediction_diff = 2 * (m * x_i + b - y_i)
        gradient = np.array([[prediction_diff], [prediction_diff * x_i]])
        gradients += gradient
    return gradients


def gradient_LSE_lnalg(weights, X_batch, y_batch):
    """Return the gradient of LSE loss function with "sTyLe".
    Return: np.array[[change in y-intercept], [change in slope]]
    """
    # https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/13/lecture-13.pdf
    return 2 * X_batch.T @ (X_batch @ weights - y_batch)


def sgd(X, y, *, lr, epochs, batch_size, momentum=0):
    X_size = len(X)
    # [[b], [m]]
    weights = betas.copy()

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

            gradient = gradient_LSE_lnalg(weights, X_batch, y_batch)
            if i > 0:
                velocity = momentum * velocity + gradient
            else:
                velocity = gradient

            weights -= lr * velocity / batch_size

        # [1, x] * [[b], [m]] -> [b + x * m]
        predictions = X_bias @ weights
        cost = np.mean(LSE(predictions, y))

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, cost: {cost}")
            print(f"Beta:\n{weights}")
            print("===================")
            beta_history.append(weights.copy())
            cost_history.append(cost)

    print(f"Final: b={weights[1][0]}, m={weights[0][0]}")
    predictions = X_bias @ weights
    cost = np.mean(LSE(predictions, y))
    print(f"Cost: {cost}")
    beta_history.append(weights)
    cost_history.append(cost)

    return beta_history, cost_history


def losses(X_bias, y, weights):
    """Return a column vector containing total losses.

    Parameters
    ----------
    X_bias : _type_
        An `nx2` matrix with its first column filled with 1s and second column filled with all possible values of x.
    y : _type_
        A `nx1` vector containing all expected value of the function.
    weights : _type_
        A `2xm` matrix containing parameters to the function.

    Returns
    -------
    _type_
        Return a column vector containing total losses.
    """
    # A matrix containing all possible losses for all values of (X, y) and (m, b).
    loss_matrix = (y - (X_bias @ weights)) ** 2
    # Find 2-norm of the matrix.
    # Essentially "collapse" the matrix MxN into 1xN using sum(abs(matrix[i][c]), i=0->M)
    all_losses = np.linalg.norm(loss_matrix, axis=0, ord=2)
    return all_losses


def plot3d(*, center_on, beta_history, spanning_radius=6):
    b, m = center_on
    b_range = np.arange(b - spanning_radius, b + spanning_radius, 0.05)
    m_range = np.arange(m - spanning_radius, m + spanning_radius, 0.05)
    b_grid, m_grid = np.meshgrid(b_range, m_range)

    range_len = len(b_range)
    # Make [b, m] in (2, 14400) shape
    all_bm_values = np.hstack([b_grid.flatten()[:, None], m_grid.flatten()[:, None]]).T

    # Compute all losses, reshape back to grid format
    all_losses = losses(X_bias, Y, all_bm_values).reshape((range_len, range_len))

    xs = []
    ys = []
    for betas in beta_history:
        b = betas[0][0]
        m = betas[1][0]
        xs.append(b)
        ys.append(m)

    # Graph loss function.
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(b_grid, m_grid, all_losses, alpha=0.5, cmap="RdBu")

    # Plot some of the chosen beta values.
    selected_points = losses(X_bias, Y, np.column_stack(beta_history))
    color_first = np.array([0, 0, 0, 1])
    color_last = np.array([1, 0, 0, 1])
    color_between = np.random.rand(len(selected_points) - 2, 4)
    colors = np.stack((color_first, *color_between, color_last))

    markers = ["x"] + ["o"] * (len(selected_points) - 2) + ["s"]

    for x, y, z, color, style in zip(
        xs, ys, selected_points, colors, markers, strict=True
    ):
        ax.scatter(x, y, z, color=color, marker=style)

    ax.set_xlabel("b")
    ax.set_ylabel("m")
    ax.set_zlabel("Cost")
    ax.set_xticks(np.arange(b - spanning_radius, b + spanning_radius, 2))
    ax.set_yticks(np.arange(m - spanning_radius, m + spanning_radius, 2))
    ax.set_zticks([])
    plt.show()


def plot_bestfit(X, y, beta_final):
    plt.scatter(X, y, color="blue", label="Data points")
    plt.plot(
        X,
        X_bias @ beta_final,
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
        Y,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        momentum=MOMENTUM,
    )

    plot3d(center_on=(4, 3), beta_history=beta_history, spanning_radius=6)
