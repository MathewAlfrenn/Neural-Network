import os
import numpy as np
from model_utils import load_model
from neural_net import NeuralNetwork

def initialize_network():
    """Initialize the neural network and load the pre-trained model."""
    input_size = 784
    hidden_layers = [512, 512]
    output_size = 10
    nn = NeuralNetwork(input_size, hidden_layers, output_size)
    load_model(nn, "model.npz")
    
    #print(f"Weights shape: {[w.shape for w in nn.weights]}")  # Print weights shape
    #print(f"Biases shape: {[b.shape for b in nn.biases]}")  # Print biases shape
    
    return nn


def predict_digit(nn, image):
    """Given a 28x28 image, predict the digit using the trained neural network."""
    image = np.array(image).reshape(1, 784)  # Reshape to match the network input
    output = nn.forward(image)
    predicted_label = np.argmax(output)  # Get the index of the max output (predicted class)
    return predicted_label

# Main function for testing (if needed)
def main():
    nn = initialize_network()

    # You can include code here to test or load additional data
    # This is just for testing predictions on a single image
    test_image = np.random.rand(28, 28)  # Example random image
    prediction = predict_digit(nn, test_image)
    print(f"Predicted Label: {prediction}")

if __name__ == "__main__":
    main()
