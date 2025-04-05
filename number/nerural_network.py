import numpy as np  
from os.path import join



# Helper functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def format_output_probabilities(output):
    percentages = output * 100  
    predicted_number = np.argmax(output, axis=1)[0]
    
    print("\nNumber Probabilities (%):")
    for number, percentage in enumerate(percentages[0]):
        print(f"Number {number}: {percentage:.2f}%")
    
    print(f"\nPredicted Number: {predicted_number}")

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-9)
    return np.sum(log_likelihood) / m

# Neural Network Class 
class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))

        # Last hidden layer to output
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):
        self.layers = [inputs]
        
        for i in range(len(self.weights)):
            z = np.dot(self.layers[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                a = softmax(z)  # Output layer
            else:
                a = relu(z)  # Hidden layers
            self.layers.append(a)

        return self.layers[-1]

    def backpropagate(self, inputs, labels, epoch, initial_lr=0.01, decay_rate=0.01):
    # Get the current learning rate based on the epoch
        #learning_rate = self.learning_rate_schedule(initial_lr, epoch, decay_rate)
        learning_rate = initial_lr
        outputs = self.forward(inputs)
        loss = cross_entropy_loss(labels, outputs)

        dL_dOut = outputs - labels
        gradients_weights = []
        gradients_biases = []

        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layers[i].T, dL_dOut) / inputs.shape[0]
            dB = np.sum(dL_dOut, axis=0, keepdims=True) / inputs.shape[0]
            gradients_weights.insert(0, dW)
            gradients_biases.insert(0, dB)

            if i > 0:
                dL_dOut = np.dot(dL_dOut, self.weights[i].T) * relu_derivative(self.layers[i])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_weights[i]
            self.biases[i] -= learning_rate * gradients_biases[i]

        return loss
    def backpropagate(self, inputs, labels, learning_rate=0.01):
        outputs = self.forward(inputs)
        loss = cross_entropy_loss(labels, outputs)

        dL_dOut = outputs - labels
        gradients_weights = []
        gradients_biases = []

        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layers[i].T, dL_dOut) / inputs.shape[0]
            dB = np.sum(dL_dOut, axis=0, keepdims=True) / inputs.shape[0]
            gradients_weights.insert(0, dW)
            gradients_biases.insert(0, dB)

            if i > 0:
                dL_dOut = np.dot(dL_dOut, self.weights[i].T) * relu_derivative(self.layers[i])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_weights[i]
            self.biases[i] -= learning_rate * gradients_biases[i]

        return loss

    def learning_rate_schedule(self, initial_lr, epoch, decay_rate=0.01):
        return initial_lr * np.exp(-decay_rate * epoch)