from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import io

# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model_path = "/home/gust141999/capstone/Model"
loaded_model = tf.keras.models.load_model(model_path)

# Define class names
class_names = [
    'Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust', 
    'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic', 
    'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight', 'Rice___Brown_Spot', 'Rice___Healthy', 
    'Rice___Hispa', 'Rice___Leaf_Blast', 'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust'
]

# Define a function to load an image with error handling
def load_image(image_path):
    try:
        image_stream = io.BytesIO(image_path.read())
        # Attempt to load the image
        image = tf.keras.preprocessing.image.load_img(image_stream, target_size=(224, 224))
        return image
    except (tf.keras.preprocessing.image.ImageDecodeException, FileNotFoundError) as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


# Function to convert image to prediction format
def img_to_pred(image):
    return tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image), axis=0)

# Endpoint to receive images and send predictions
@app.route('/predict', methods=['POST'])
def predict_image():
    # Receive image
    image_file = request.files['image']
    image = load_image(image_file)
    
    # Convert image to prediction format
    image_array = img_to_pred(image)

    if image is not None:
        # Make predictions
        predictions = loaded_model.predict(image_array)
        
        # Get the predicted class
        predicted_class_index = tf.argmax(predictions[0]).numpy()
        predicted_class_name = class_names[predicted_class_index]

        # Return prediction
        return jsonify({'prediction': predicted_class_name}), 200
    else:
        return jsonify({'error': 'Image loading failed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
