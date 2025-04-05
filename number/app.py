import numpy as np
from flask import Flask, render_template, request, jsonify
import main  # Import the neural network module
import matplotlib.pyplot as plt

app = Flask(__name__)

# Initialize the neural network
nn = main.initialize_network()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.json['image']
    print_grid_from_received_data(data)

    # Convert data to a NumPy array and reshape to (28, 28)
    image_data = np.array(data, dtype=np.float32).reshape(28, 28)

    # Use the predict function
    predicted_label = main.predict_digit(nn, image_data)

    return jsonify({'prediction': int(predicted_label)})
def print_grid_from_received_data(received_data):
    """
    Convert the received 4-dimensional data into a 28x28 grid and print it.

    Parameters:
    - received_data: A 4-dimensional list containing pixel values (in shape (28, 28, 1)).

    This function prints the grid in a human-readable form.
    """
    # Convert the received data into a NumPy array for easier manipulation
    data_array = np.array(received_data)

    # Remove the unnecessary extra dimension (shape will become (28, 28))
    data_array = np.squeeze(data_array)

    # Convert values to 0 and 1, where any non-zero value is considered white (1), and 0 is black (0)
    grid = np.where(data_array > 0, 1, 0)

    # Print the 28x28 grid
    for row in grid:
        print(' '.join(map(str, row)))


if __name__ == '__main__':
    app.run(debug=True)
