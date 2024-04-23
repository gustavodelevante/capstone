import requests

# Define the URL of the Flask app on Raspberry Pi
raspberry_pi_ip = "10.3.43.130"
flask_port = "5000"
endpoint_url = f"http://{raspberry_pi_ip}:{flask_port}/predict"

# Load image
image_path = r"C:\Users\gusta\OneDrive\Escritorio\Capstone\Crops Data\Cashew_only\Potato___Early_Blight\0a6983a5-895e-4e68-9edb-88adf79211e9___RS_Early.B 9072.JPG"

# Open the image file
image_file = open(image_path, 'rb')

# Send HTTP POST request with the image
response = requests.post(endpoint_url, files={'image': image_file})

# Close the file
image_file.close()

# Check if the request was successful
if response.status_code == 200:
    # Parse response
    prediction = response.json().get('prediction')
    
    if prediction is not None:
        print("Predicted class:", prediction)
    else:
        print("Error: Prediction not received")
else:
    print("Error:", response.text)
