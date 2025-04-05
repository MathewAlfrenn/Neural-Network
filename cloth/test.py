import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

from dataset_loader import MnistDataloader
from nerural_network import NeuralNetwork
from model_utils import labels_encode, save_model, load_model

def main():
    # Set file paths based on MNIST dataset location
    input_path = os.path.join(os.path.dirname(__file__), "archive")

    # Define file paths for the dataset
    training_images_filepath = os.path.join(input_path, "train-images-idx3-ubyte")
    training_labels_filepath = os.path.join(input_path, "train-labels-idx1-ubyte")
    test_images_filepath = os.path.join(input_path, "t10k-images-idx3-ubyte")
    test_labels_filepath = os.path.join(input_path, "t10k-labels-idx1-ubyte")

    # Load the dataset
    data_loader = MnistDataloader(training_images_filepath, training_labels_filepath, 
                                  test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    # Initialize the neural network with input size, hidden layers, and output size
    input_size = 784
    hidden_layers = [512, 512]
    output_size = 10
    nn = NeuralNetwork(input_size, hidden_layers, output_size)

    # Load the trained model if it exists (to avoid retraining)
    load_model(nn, "model.npz")

    # Testing loop
    correct_predictions = 0
    total_predictions = len(x_test)
    misclassified_images = []

    for i in range(total_predictions):
        # Reshape the test image to match the input shape of the neural network
        sample_input = np.array(x_test[i]).reshape(1, 784)
        output = nn.forward(sample_input)
        
        # Get the predicted label from the output
        predicted_label = np.argmax(output)
        
        # Check if the prediction is correct
        if predicted_label == y_test[i]:
            correct_predictions += 1
        else:
            misclassified_images.append((x_test[i], y_test[i], predicted_label))

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Store and print misclassified images in groups of 16
    if misclassified_images:
        print(f"Total misclassified images: {len(misclassified_images)}")
        # Display misclassified images in groups of 16
        images_per_row = 4  # 4 columns per row
        images_per_column = 4  # 4 rows per column, this gives a 4x4 grid
        total_images = len(misclassified_images)
        total_groups = total_images // (images_per_row * images_per_column)
        if total_images % (images_per_row * images_per_column) != 0:
            total_groups += 1

        for group in range(total_groups):
            plt.figure(figsize=(10, 10))  # Increase the figure size
            start_idx = group * images_per_row * images_per_column
            end_idx = min((group + 1) * images_per_row * images_per_column, total_images)

            for idx, (image, true_label, predicted_label) in enumerate(misclassified_images[start_idx:end_idx]):
                ax = plt.subplot(images_per_row, images_per_column, idx + 1)
                ax.imshow(image, cmap=plt.cm.gray)
                ax.set_title(f"True: {true_label}, Pred: {predicted_label}")
                ax.axis('off')  # Hide axis

            plt.tight_layout()  # Adjust layout to avoid overlap
            #plt.show()  # Uncomment this to actually display the plots

if __name__ == "__main__":
    main()
