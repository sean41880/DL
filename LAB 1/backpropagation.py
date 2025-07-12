import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100, seed=None):
    import numpy as np
    np.random.seed(seed)
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs  = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1 - 0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)
               
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2 / hidden1_size)
    b2 = np.zeros((1, hidden2_size))
    W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2 / hidden2_size)
    b3 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3

def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    #A1 = relu(Z1)
    #A1 = Z1
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    #A2 = relu(Z2)
    #A2 = Z2
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    #A3 = relu(Z3)
    #A3 = Z3
    return Z1, A1, Z2, A2, Z3, A3

def backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3):
    m = X.shape[0]
    dZ3 = A3 - Y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    dZ2 = np.dot(dZ3, W3.T) * sigmoid_derivative(Z2)
    #dZ2 = np.dot(dZ3, W3.T) * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)
    #dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W2, b2, W3, b3

def train(X, Y, input_size, hidden1_size, hidden2_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2, W3, b3 = initialize_parameters(input_size, hidden1_size, hidden2_size, output_size)
    losses = []
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
        loss = np.mean((A3 - Y) ** 2)
        losses.append(loss)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3)
        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    for i in range(len(X)):
        ground_truth = Y[i, 0]
        prediction = A3[i, 0]
        print(f"Iter{i+1} | Ground truth: {ground_truth:.1f} | Prediction: {prediction:.5f}")

    return W1, b1, W2, b2, W3, b3, losses

def calculate_accuracy(Y_true, Y_pred):
    Y_pred_binary = (Y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    accuracy = np.mean(Y_true == Y_pred_binary) * 100
    return accuracy

def plot_comparison(X, Y_true, Y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=Y_true.flatten(), cmap='coolwarm', label='Ground Truth', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=Y_pred.flatten(), cmap='coolwarm', marker='x', label='Predictions', alpha=0.6)
    plt.title("Comparison of Predictions and Ground Truth")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

def plot_decision_boundary(X, Y, W1, b1, W2, b2, W3, b3):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    _, _, _, _, _, A3 = forward_propagation(grid, W1, b1, W2, b2, W3, b3)
    Z = (A3 > 0.5).astype(int).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolor='k', cmap='coolwarm')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_results(losses):
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

if __name__ == "__main__":
    x, y = generate_linear(n=100, seed = 42)

    input_size = 2
    hidden1_size = 4
    hidden2_size = 3
    output_size = 1
    epochs = 100000
    learning_rate = 0.01

    W1, b1, W2, b2, W3, b3, losses = train(x, y, input_size, hidden1_size, hidden2_size, output_size, epochs, learning_rate)

    plot_results(losses)

    _, _, _, _, _, predictions = forward_propagation(x, W1, b1, W2, b2, W3, b3)

    accuracy = calculate_accuracy(y, predictions)
    print(f"Accuracy: {accuracy:.2f}%")

    plot_comparison(x, y, predictions)

    plot_decision_boundary(x, y, W1, b1, W2, b2, W3, b3)