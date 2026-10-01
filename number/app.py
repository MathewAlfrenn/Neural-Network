import numpy as np
from flask import Flask, render_template, request, jsonify
import main  # Import the neural network module
import matplotlib.pyplot as plt
from PIL import Image

def center_like_mnist(img):
    ys, xs = np.nonzero(img > 0)
    if len(ys) == 0:
        return img
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.array(
        Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR),
        dtype=np.float32,
    )
    out = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    out[top:top + new_h, left:left + new_w] = small
    total = out.sum()
    if total > 0:
        cy = (out.sum(axis=1) * np.arange(28)).sum() / total
        cx = (out.sum(axis=0) * np.arange(28)).sum() / total
        out = np.roll(out, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))
    return out

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
    image_data = image_data * 255 
    image_data = center_like_mnist(image_data)
    
    #print(image_data)
    # Flatten
    image_data = image_data.reshape(1, 784) #predict alr does that

    # Use the predict function
    predicted_label = main.predict_digit(nn, image_data)

    return jsonify({'prediction': int(predicted_label)})
def print_grid_from_received_data(received_data):
    """
    Convert the received 4-dimensional data into a 28x28 grid and print it.

    Parameters:
    - received_data: A 4-dimensional list containing pixel values (in shape (28, 28, 1)).

    This function prints the grid in a human-readable form as integers.
    """
    # Convert the received data into a NumPy array for easier manipulation
    data_array = np.array(received_data, dtype=np.float32)

    # Remove the unnecessary extra dimension (shape will become (28, 28))
    data_array = np.squeeze(data_array)

    # Convert values to 0 or 255 (integer)
    grid = np.where(data_array > 0, 255, 0).astype(int)

    # Print the 28x28 grid as integers
    for row in grid:
        print(' '.join(map(str, row)))



if __name__ == '__main__':
    app.run(debug=True)
