import firebase_admin
from firebase_admin import credentials, storage
import os
import time
import torch
from torchvision import transforms
from PIL import Image
import io

# Firebase initialization
cred = credentials.Certificate('crop-diagnosis-detector-firebase-adminsdk-1sdc6-87adfecacb.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'crop-diagnosis-detector.appspot.com'
})

bucket = storage.bucket()

# Directory to save images
IMAGE_SAVE_PATH = 'firebase_listener_images'  # Make sure this directory exists

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the PyTorch model (you may need to adjust this if using a different architecture)
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

# Load the PyTorch model
model = CustomModel().to(device)
model.load_state_dict(torch.load("/home/gust141999/my_project/Model_best_v23.pth", map_location=device, weights_only=True))
model.eval()

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

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to list all files in the specific folder in Firebase Storage
def list_files_in_folder(folder_name):
    blobs = bucket.list_blobs(prefix=folder_name)
    return blobs

# Function to download a file from Firebase Storage
def download_image(blob, file_name):
    file_path = os.path.join(IMAGE_SAVE_PATH, file_name)
    
    # Download the file if it doesn't already exist
    if not os.path.exists(file_path):
        print(f"Downloading: {file_name}")
        blob.download_to_filename(file_path)
        print(f"Downloaded successfully: {file_name}")
    else:
        print(f"File {file_name} already exists. Skipping download.")
    return file_path

# Function to load and preprocess image for prediction
def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to upload the diagnosis result to Firebase Storage
def upload_to_firebase(userID, imageStorageName, prediction):
    try:
        # Remove the .jpg extension if present
        if imageStorageName.endswith('.jpg'):
            imageStorageName = os.path.splitext(imageStorageName)[0]

        # Define the file path in Firebase Storage for the diagnosis
        destination_blob_name = f"{userID}/diagnoses/{imageStorageName}_diagnosis.txt"

        # Create a diagnosis file content with the prediction
        diagnosis_text = f"Diagnosis: {prediction}\n"
        
        # Save the diagnosis text temporarily on the local system
        temp_filename = f"{imageStorageName}_diagnosis.txt"
        with open(temp_filename, "w") as f:
            f.write(diagnosis_text)

        # Upload the diagnosis text file to Firebase Storage
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(temp_filename)

        # Log successful upload
        print(f"Uploaded diagnosis to {destination_blob_name}")

        # Cleanup the local diagnosis file
        os.remove(temp_filename)

    except Exception as e:
        print(f"Error uploading to Firebase: {e}")

# Poll Firebase Storage for new images
def poll_firebase_storage():
    folder_name = 'new_images/'  # The folder you want to monitor

    while True:
        print("Checking for new files...")
        blobs = list_files_in_folder(folder_name)

        for blob in blobs:
            file_name = blob.name.split('/')[-1]  # Get just the file name

            if file_name not in downloaded_files:
                try:
                    # Split the file name by '&' delimiter to get userID and imageStorageName
                    user_id, image_storage_name = file_name.split('&', 1)
                    print(f"New file found: {file_name}")
                    print(f"userID: {user_id}, imageStorageName: {image_storage_name}")
                    
                    # Download the image
                    image_path = download_image(blob, file_name)
                    
                    # Load and preprocess the image for prediction
                    image_tensor = load_image(image_path)

                    if image_tensor is not None:
                        # Move the image tensor to the correct device
                        image_tensor = image_tensor.to(device)

                        # Make predictions
                        with torch.no_grad():
                            outputs = model(image_tensor)
                            _, predicted_class = torch.max(outputs, 1)
                            predicted_class_name = class_names[predicted_class.item()]

                        # Print prediction to console
                        print(f"Prediction: {predicted_class_name}")

                        # Upload the diagnosis (prediction result) to Firebase Storage
                        upload_to_firebase(user_id, image_storage_name, predicted_class_name)
                        
                    # Add the file to the list of downloaded files
                    downloaded_files.add(file_name)
                    save_downloaded_files()  # Update the list of downloaded files

                except ValueError:
                    # Handle cases where the file name doesn't match the expected pattern
                    print(f"Skipping file {file_name} - does not contain the expected delimiter '&'")
                    continue

        # Wait before checking again
        time.sleep(1)  # Poll every 1 second

# Store previously downloaded files to avoid duplicates
downloaded_files = set()

# Load previously downloaded files from a file (persistent state)
def load_downloaded_files():
    global downloaded_files
    if os.path.exists("downloaded_files.txt"):
        with open("downloaded_files.txt", "r") as file:
            downloaded_files = set(line.strip() for line in file)
        print("Loaded previously downloaded files.")
    else:
        print("No previously downloaded files found.")

# Save downloaded files to a file for persistence
def save_downloaded_files():
    with open("downloaded_files.txt", "w") as file:
        for file_name in downloaded_files:
            file.write(f"{file_name}\n")
    print("Saved downloaded files.")

if __name__ == "__main__":
    try:
        load_downloaded_files()  # Load previously downloaded files
        poll_firebase_storage()
    except KeyboardInterrupt:
        print("Stopping polling...")
        save_downloaded_files()  # Save downloaded files on exit
