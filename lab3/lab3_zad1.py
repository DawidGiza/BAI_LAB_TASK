import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def mlp(x, w1, w2, beta=1.0):
    """
    Forward pass MLP:

    x   - input vector [x1, x2]
    w1  - weights input -> hidden (shape: 2 x 3)  (2 neurons, 3 inputs with bias)
    w2  - weights hidden -> output (shape: 1 x 3) (1 neuron, 3 inputs with bias)
    beta - optional scaling parameter (not strictly required, but included per instruction)
    """

    x_bias = np.array([1.0, x[0], x[1]])

    net_hidden = np.dot(w1, x_bias)
    v_hidden = tanh(beta * net_hidden)     

    v_bias = np.concatenate(([1.0], v_hidden)) 

    net_output = np.dot(w2, v_bias)
    y = sigmoid(beta * net_output)

    return {
        "output": y,
        "x_bias": x_bias,
        "net_hidden": net_hidden,
        "v_hidden": v_hidden,
        "v_bias": v_bias,
        "net_output": net_output
    }

x = np.array([0.5, -0.2])

w1 = np.array([
    [0.2, -0.4, 0.1],
    [-0.3, 0.8, -0.5]
])

w2 = np.array([[0.7, -1.2, 0.3]])

result = mlp(x, w1, w2)

print("Output:", result["output"])
print("Hidden layer:", result["v_hidden"])