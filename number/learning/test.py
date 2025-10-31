import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_loader import MnistDataloader
from neural_net import NeuralNetwork
from model_utils import load_model

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
    (_, _), (x_test, y_test) = data_loader.load_data()

    # Initialize the neural network
    input_size = 784
    hidden_layers = [512, 512]
    output_size = 10
    nn = NeuralNetwork(input_size, hidden_layers, output_size)

    # Load the trained model
    load_model(nn, "model.npz")

    # Testing loop
    correct_predictions = 0
    total_predictions = len(x_test)
    misclassified_images = []

    for i in range(total_predictions):
        sample_input = np.array(x_test[i]).reshape(1, 784)
        output = nn.forward(sample_input)
        predicted_label = np.argmax(output)

        # Show the first test image for verification
        if i == 0:
            print("**TEST SAMPLE**")
            print("Correct Label:", y_test[i])
            print(sample_input)
            plt.imshow(x_test[i], cmap='gray')
            plt.title(f"Correct Label: {y_test[i]}")
            plt.show()
            print("Predicted Label:", predicted_label)

        # Count correct and incorrect predictions
        if predicted_label == y_test[i]:
            correct_predictions += 1
        else:
            misclassified_images.append((x_test[i], y_test[i], predicted_label))

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Display misclassified images
    if misclassified_images:
        print(f"Total misclassified images: {len(misclassified_images)}")
        images_per_row = 4
        images_per_column = 4
        total_images = len(misclassified_images)
        total_groups = (total_images + (images_per_row * images_per_column) - 1) // (images_per_row * images_per_column)

        for group in range(total_groups):
            plt.figure(figsize=(10, 10))
            start_idx = group * images_per_row * images_per_column
            end_idx = min((group + 1) * images_per_row * images_per_column, total_images)

            for idx, (image, true_label, pred_label) in enumerate(misclassified_images[start_idx:end_idx]):
                ax = plt.subplot(images_per_row, images_per_column, idx + 1)
                ax.imshow(image, cmap='gray')
                ax.set_title(f"True: {true_label}, Pred: {pred_label}")
                ax.axis('off')

            plt.tight_layout()
            # plt.show()  # Uncomment if you want to see them

if __name__ == "__main__":
    main()
