# The general structure of this file is borrowed from:
# https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(seed=471)

# Data as vectors. X is input vector, Y is expected vector.
X = 2 * rng.random((100, 1))
Y = 4 + 3 * X + rng.standard_normal((100, 1))

# From one of the lecture example. Warning: the graph won't look nice because not enough (X, Y).
# X = np.array([[0], [1], [2]])
# Y = np.array([[4], [3], [1]])

# Configure SGD:
LEARNING_RATE = 0.0001  # Between 0 and 0.1. The smaller the slower it converges.
EPOCHS = 1000  # Around 5000 max is enough, no need to go crazy.
BATCH_SIZE = 10
MOMENTUM = 0.9  # Between 0 and 1; 0 is no momentum. Don't set momentum too high when learning rate is high.

# Configure graph:
CENTER_ON = (4, 3)  # Should be on the expected value.
EPOCH_PER_POINT = 100  # Number of points to plot = EPOCHS / EPOCH_PER_POINT.
CONTOUR_LAYOUT = np.concat(
    (
        np.arange(0, 200, 50),
        np.arange(200, 500, 100),
        np.arange(500, 10000, 1000),
    )
)  # np.arange(start, stop, step)


# We'd like to find m and b that best fit this data.

# Don't change anything below.

# The regressor matrix. Convenient to have it here cuz we'll use it throughout.
X_bias = np.c_[np.ones((len(X), 1)), X]


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
    # https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
    X_size = len(X)
    # [[b], [m]]
    weights = rng.standard_normal((2, 1))

    cost_history = []
    beta_history = []

    for epoch in range(epochs):
        # Move the rows only, keep the column the same.
        indices = rng.permutation(X_size)
        X_shuffle = X_bias[indices]
        y_shuffle = y[indices]
        velocity = 0

        # For each epoch, we go through all points at least once.
        for i in range(0, X_size, batch_size):
            X_batch = X_shuffle[i : i + batch_size]
            y_batch = y_shuffle[i : i + batch_size]

            # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
            # Changed to simple momentum formula because dampening seems overkill.
            gradient = gradient_LSE_lnalg(weights, X_batch, y_batch)
            if i > 0:
                velocity = momentum * velocity + gradient
            else:
                velocity = gradient
            
            weights -= lr / batch_size * velocity


        # [1, x] * [[b], [m]] -> [b + x * m]
        predictions = X_bias @ weights
        cost = np.mean(LSE(predictions, y))
        beta_history.append(weights.copy())
        cost_history.append(cost)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, cost: {cost}")
            print(f"Beta:\n{weights}")
            print("===================")

    print(f"Final: m={weights[1][0]}, b={weights[0][0]}")
    predictions = X_bias @ weights
    cost = np.mean(LSE(predictions, y))
    print(f"Cost: {cost}")

    return beta_history, cost_history


def losses(X_bias, y, weights):
    """Return a column vector containing total losses.

    Parameters
    ----------
    X_bias : NDArray
        An `nx2` matrix with its first column filled with 1s and second column filled with all possible values of x.
    y : NDArray
        A `nx1` vector containing all expected value of the function.
    weights : NDArray
        A `2xm` matrix containing parameters to the function.

    Returns
    -------
    NDArray
        Return a column vector containing total losses.
    """
    # https://aunnnn.github.io/ml-tutorial/html/blog_content/linear_regression/linear_regression_tutorial.html

    # A matrix containing all possible losses for all values of (X, y) and (m, b).
    loss_matrix = y - (X_bias @ weights)
    # Find 2-norm of the matrix along the column.
    # This norm is calculated by sum(abs(matrix[i][c]) ** 2, i=0->n) ** (1/2)
    # Since this operation conveniently sum up all the loss for a particular (b, m), we want to
    # undo the sqrt at the end, hence ** 2.
    all_losses = np.linalg.norm(loss_matrix, axis=0, ord=2) ** 2
    return all_losses


def get_points_info(length: int):
    """Return colors, markers, and size info for `length` points."""

    color_first = np.array([0, 0, 0, 1])  # Black
    color_last = np.array([1, 0, 0, 1])  # Red
    color_between = np.c_[rng.random((length - 2, 3)), np.ones((length - 2, 1))]
    # Unpack color_between because it is unnecessary nested.
    colors = np.stack((color_first, *color_between, color_last))
    markers = ["8"] + ["o"] * (length - 2) + ["*"]
    sizes = [50] + [15] * (length - 2) + [50]

    return colors, markers, sizes


def plot3d(*, center_on, beta_history, spanning_radius=6):
    beta_history = beta_history[::EPOCH_PER_POINT]
    b, m = center_on
    b_range = np.arange(b - spanning_radius, b + spanning_radius, 0.05)
    m_range = np.arange(m - spanning_radius, m + spanning_radius, 0.05)
    b_grid, m_grid = np.meshgrid(b_range, m_range)

    # https://aunnnn.github.io/ml-tutorial/html/blog_content/linear_regression/linear_regression_tutorial.html
    range_len = len(b_range)
    # Make [b, m] in (2, n) shape
    all_bm_values = np.hstack([b_grid.flatten()[:, None], m_grid.flatten()[:, None]]).T

    # Compute all losses, reshape back to grid format
    all_losses = losses(X_bias, Y, all_bm_values).reshape((range_len, range_len))

    # There's probably a better way to do this but it works and it's readable so :/
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

    # Plot some of the chosen beta values.
    selected_points_z = losses(X_bias, Y, np.column_stack(beta_history))
    colors, markers, sizes = get_points_info(len(beta_history))
    for x, y, z, color, style, size in zip(
        xs, ys, selected_points_z, colors, markers, sizes, strict=True
    ):
        ax.scatter(x, y, z, sizes=(size,), color=color, marker=style)

    # Plot it after the points so it's easier to see the points.
    ax.plot_surface(b_grid, m_grid, all_losses, alpha=0.5, cmap="RdBu")

    ax.set_xlabel("b")
    ax.set_ylabel("m")
    ax.set_zlabel("Cost")
    ax.set_xticks(np.arange(b - spanning_radius, b + spanning_radius, 2))
    ax.set_yticks(np.arange(m - spanning_radius, m + spanning_radius, 2))
    ax.set_zticks([])
    plt.show()


def plot_contour(*, center_on, beta_history, spanning_radius=6):
    beta_history = beta_history[::EPOCH_PER_POINT]
    b, m = center_on
    b_range = np.arange(b - spanning_radius, b + spanning_radius, 0.05)
    m_range = np.arange(m - spanning_radius, m + spanning_radius, 0.05)
    b_grid, m_grid = np.meshgrid(b_range, m_range)

    # https://aunnnn.github.io/ml-tutorial/html/blog_content/linear_regression/linear_regression_tutorial.html
    range_len = len(b_range)
    # Make [b, m] in (2, n) shape
    all_bm_values = np.hstack([b_grid.flatten()[:, None], m_grid.flatten()[:, None]]).T

    # Compute all losses, reshape back to grid format
    all_losses = losses(X_bias, Y, all_bm_values).reshape((range_len, range_len))

    # There's probably a better way to do this but it works and it's readable so :/
    xs = []
    ys = []
    for betas in beta_history:
        b = betas[0][0]
        m = betas[1][0]
        xs.append(b)
        ys.append(m)

    fig, ax = plt.subplots(figsize=(10, 6))
    cs = ax.contour(
        b_grid,
        m_grid,
        all_losses,
        levels=CONTOUR_LAYOUT,
    )
    ax.clabel(cs, cs.levels, fmt=lambda x: f"{x:.0f}", fontsize=10)

    colors, markers, sizes = get_points_info(len(beta_history))
    # Plot lines connecting the points to show the order of traversal.
    ax.plot(xs, ys, color="red", linestyle="dotted")
    # Plotting points with different marker styles is suprisingly annoying...
    for i, (x, y, color, style, size) in enumerate(
        zip(xs, ys, colors, markers, sizes, strict=True)
    ):
        ax.scatter(x, y, sizes=(size,), color=color, marker=style)

    ax.set_xlabel("b")
    ax.set_ylabel("m")
    ax.set_xticks(np.arange(b - spanning_radius, b + spanning_radius, 2))
    ax.set_yticks(np.arange(m - spanning_radius, m + spanning_radius, 2))
    plt.show()


def plot_bestfit(X, y, beta_final):
    # https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
    plt.scatter(X, y, color="blue", label="Data points")
    plt.plot(
        X,
        X_bias @ beta_final,
        color="red",
        label=f"y = {beta_final[0][0]:.2f} + {beta_final[1][0]:.2f}x",
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

    plot_bestfit(X, Y, beta_history[-1])
    plot3d(center_on=CENTER_ON, beta_history=beta_history, spanning_radius=6)
    plot_contour(center_on=CENTER_ON, beta_history=beta_history, spanning_radius=6)
