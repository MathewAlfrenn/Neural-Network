import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_loader import MnistDataloader
from neural_net import NeuralNetwork
from model_utils import labels_encode, save_model, load_model

def main():
    # Set file paths based on MNIST dataset location
    input_path = "./archive"  

    training_images_filepath = os.path.join(input_path, "train-images.idx3-ubyte")
    training_labels_filepath = os.path.join(input_path, "train-labels.idx1-ubyte")
    test_images_filepath = os.path.join(input_path, "t10k-images.idx3-ubyte")
    test_labels_filepath = os.path.join(input_path, "t10k-labels.idx1-ubyte")

    # Load the dataset
    data_loader = MnistDataloader(training_images_filepath, training_labels_filepath, 
                                  test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    # Initialize the neural network
    input_size = 784
    hidden_layers = [512, 512]
    output_size = 10
    nn = NeuralNetwork(input_size, hidden_layers, output_size)

    # Load the model if it exists (this is key to not resetting weights each time)
    load_model(nn, "model.npz")

    # Training parameters
    epochs = 10
    learning_rate = 0.01
    batch_size = 64

    # Training loop
    for epoch in range(epochs):
        loss_epoch = 0
        for i in range(0, len(x_train), batch_size):
            # Handle case where the last batch may be smaller than batch_size
            batch_x = np.array(x_train[i:i+batch_size]).reshape(-1, 784)  # Reshape to (batch_size, 784)
            batch_y = labels_encode(y_train[i:i+batch_size], 10)
            loss = nn.backpropagate(batch_x, batch_y, learning_rate)
            loss_epoch += loss
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_epoch / (len(x_train) / batch_size):.4f}")

    # Save the model after training
    save_model(nn, "model.npz")

    # Select a random test image
    index = random.randint(0, len(x_test) - 1)
    plt.imshow(x_test[index], cmap=plt.cm.gray)
    plt.title(f"Actual Label: {y_test[index]}")
    plt.show()

    # Predict using the trained model
    sample_input = np.array(x_test[index]).reshape(1, 784)
    output = nn.forward(sample_input)
    predicted_label = np.argmax(output)
    print(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()