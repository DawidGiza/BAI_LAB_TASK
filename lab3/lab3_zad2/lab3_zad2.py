

import matplotlib.pyplot as plt

xx = np.array([
    [1, -1, -1],
    [1, -1,  1],
    [1,  1, -1],
    [1,  1,  1]
])

d = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_diff(y, beta):
    return beta * y * (1 - y)

def tanh(x):
    return np.tanh(x)

def tanh_diff(y, beta):
    return beta * (1 - y * y)

def forward(x, W1, W2, beta):
    x_bias = x

    net_h = W1 @ x_bias
    h = tanh(beta * net_h)

    v = np.concatenate(([1.0], h))
    net_o = W2 @ v
    y = sigmoid(beta * net_o)

    return x_bias, net_h, h, v, net_o, y

def train_sample(xx, d, eta=0.5, beta=1.0):
    np.random.seed(0)

    W1 = np.random.randn(2, 3) * 0.5
    W2 = np.random.randn(1, 3) * 0.5

    errors = []

    for epoch in range(100000):

        mse = 0

        for x, target in zip(xx, d):

            x_bias, net_h, h, v, net_o, y = forward(x, W1, W2, beta)

            error = target - y
            mse += error**2

            delta_o = error * sigmoid_diff(y, beta)

            grad_W2 = delta_o * v

            delta_h = tanh_diff(h, beta) * (W2[0, 1:] * delta_o)

            grad_W1 = np.outer(delta_h, x_bias)

            W2 += eta * grad_W2
            W1 += eta * grad_W1

        errors.append(mse / len(xx))

        preds = []
        for x in xx:
            _, _, _, _, _, y = forward(x, W1, W2, beta)
            preds.append(1 if y > 0.9 else 0)

        if np.array_equal(preds, d):
            break

    return errors

def train_epoch(xx, d, eta=0.5, beta=1.0):
    np.random.seed(0)

    W1 = np.random.randn(2, 3) * 0.5
    W2 = np.random.randn(1, 3) * 0.5

    errors = []

    for epoch in range(100000):

        mse = 0

        grad_W1_sum = np.zeros_like(W1)
        grad_W2_sum = np.zeros_like(W2)

        for x, target in zip(xx, d):

            x_bias, net_h, h, v, net_o, y = forward(x, W1, W2, beta)

            error = target - y
            mse += error**2

            delta_o = error * sigmoid_diff(y, beta)
            grad_W2_sum += delta_o * v

            delta_h = tanh_diff(h, beta) * (W2[0, 1:] * delta_o)
            grad_W1_sum += np.outer(delta_h, x_bias)

        W2 += eta * grad_W2_sum
        W1 += eta * grad_W1_sum

        errors.append(mse / len(xx))

        preds = []
        for x in xx:
            _, _, _, _, _, y = forward(x, W1, W2, beta)
            preds.append(1 if y > 0.9 else 0)

        if np.array_equal(preds, d):
            break

    return errors

errors_sample = train_sample(xx, d, eta=0.5, beta=1.0)
errors_epoch = train_epoch(xx, d, eta=0.5, beta=1.0)

plt.figure(figsize=(10,5))
plt.plot(errors_sample, label="po próbce")
plt.plot(errors_epoch, label="po epoce")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()