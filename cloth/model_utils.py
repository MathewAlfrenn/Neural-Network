import os
import numpy as np

def save_model(nn, filename="model.npz"):
    """Save model weights and biases properly."""
    neural_folder = os.getcwd()  # Use current directory
    file_path = os.path.join(neural_folder, filename)

    # Save each weight and bias separately and unpacking
    data_dict = {}

    # Store weights with unique keys
    for i, w in enumerate(nn.weights):
        data_dict[f"weight_{i}"] = w

    # Store biases with unique keys
    for i, b in enumerate(nn.biases):
        data_dict[f"bias_{i}"] = b

    # Save everything
    np.savez(file_path, **data_dict)

    
    print(f"Model saved to {file_path}")

def load_model(nn, filename="model.npz"):
    """Load model weights and biases properly."""
    try:
        neural_folder = os.getcwd()
        file_path = os.path.join(neural_folder, filename)

        data = np.load(file_path, allow_pickle=True)

        # Reconstruct weights and biases from separately saved arrays
        nn.weights = [data[f"weight_{i}"] for i in range(len(nn.weights))]
        nn.biases = [data[f"bias_{i}"] for i in range(len(nn.biases))]

        print(f"Model loaded from {file_path}")
    except FileNotFoundError:
        print(f"No saved model found at {file_path}, starting fresh.")
    except KeyError as e:
        print(f"Error loading model: {e}. Check if the model was saved properly.")

def labels_encode(labels, num_classes=10):
    """Convert labels to one-hot encoded format."""
    return np.eye(num_classes)[labels] #identity matrix

def see_model(filename="model.npz"):
    """See the contents of a model file."""
    try:
        # Directly use the current directory to find the file
        neural_folder = os.getcwd()
        file_path = os.path.join(neural_folder, filename)
        
        data = np.load(file_path, allow_pickle=True)
        print(f"Contents of {file_path}:")
        
        # Print the names of the arrays (keys)
        for key in data.files:
            print(f"Key: {key}, Shape: {data[key].shape}")
        
    except FileNotFoundError:
        print(f"No saved model found at {file_path}.")
