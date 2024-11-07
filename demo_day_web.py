# Install necessary packages
# pip install flask torch torchvision firebase-admin flask-cors werkzeug pillow

from flask import Flask, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
import io
import firebase_admin
from flask_cors import CORS
from firebase_admin import credentials, storage
from werkzeug.utils import secure_filename
from collections import Counter

# Suppress irrelevant logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Firebase Admin SDK initialization
cred = credentials.Certificate("crop-diagnosis-detector-firebase-adminsdk-1sdc6-87adfecacb.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'crop-diagnosis-detector.appspot.com'
})

# Load the PyTorch model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.4),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.6),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36)  # Adjust output size as needed
        )
    
    def forward(self, x):
        return self.model(x)

# Define class names
class_names = [
    'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy', 'Apple Scab', 
    'Bell Pepper Bacterial Spot', 'Bell Pepper Healthy', 'Cashew anthracnose', 
    'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust', 
    'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 
    'Cassava healthy', 'Cassava mosaic', 'Cherry Healthy', 'Cherry Powdery Mildew', 
    'Grape Black Rot', 'Grape Esca (Black Measles)', 'Grape Healthy', 'Grape Leaf Blight', 
    'Invalid', 'Peach Bacterial Spot', 'Peach Healthy', 'Potato Early Blight', 
    'Potato Healthy', 'Potato Late Blight', 'Rice Brown Spot', 'Rice Healthy', 
    'Rice Leaf Blast', 'Strawberry Leaf Scorch', 'Strawberry Healthy', 
    'Wheat Brown Rust', 'Wheat Healthy', 'Wheat Yellow Rust'
]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the model with weights only and mapped to the appropriate device
model = CustomModel().to(device)
model.load_state_dict(torch.load("/home/gust141999/my_project/Model_best_v23.pth", map_location=device, weights_only=True))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to load and preprocess image
def load_image(image_file):
    try:
        image_stream = io.BytesIO(image_file.read())
        image = Image.open(image_stream).convert("RGB")
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Function to upload diagnosis result to Firebase Storage
def upload_to_firebase(userID, imageStorageName, prediction):
    try:
        if imageStorageName.endswith('.jpg'):
            imageStorageName = os.path.splitext(imageStorageName)[0]

        bucket = storage.bucket()
        destination_blob_name = f"{userID}/diagnoses/{imageStorageName}_diagnosis.txt"
        diagnosis_text = f"Diagnosis: {prediction}\n"
        
        temp_filename = f"{imageStorageName}_diagnosis.txt"
        with open(temp_filename, "w") as f:
            f.write(diagnosis_text)

        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(temp_filename)
        print(f"Uploaded diagnosis to {destination_blob_name}")
        os.remove(temp_filename)

    except Exception as e:
        print(f"Error uploading to Firebase: {e}")

# Endpoint to receive images, make predictions, and upload diagnosis to Firebase
@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        image_file = request.files['image']
        image_name = request.form.get('name')
        userID = request.form.get('userID')
        imageStorageName = request.form.get('imageStorageName')
        
        print(f"Received metadata: Name: {image_name}, UserID: {userID}, ImageStorageName: {imageStorageName}")
        image_tensor = load_image(image_file)

        if image_tensor is not None:
            image_tensor = image_tensor.to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted_class = torch.max(outputs, 1)
                predicted_class_name = class_names[predicted_class.item()]

            print(f"Prediction: {predicted_class_name}")
            upload_to_firebase(userID, imageStorageName, predicted_class_name)
            return jsonify({'prediction': predicted_class_name}), 200
        else:
            return jsonify({'error': 'Image loading failed'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)
