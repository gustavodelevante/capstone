import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define a function to load an image with error handling
def load_image(image_path):
    try:
        # Attempt to load the image
        image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        return image
    except keras.preprocessing.image.ImageDecodeException as e:
        # Log the error and return None for corrupted images
        print(f"Error loading image {image_path}: Invalid JPEG data or crop window.")
        return None
    except Exception as e:
        # Log the error and return None for other exceptions
        print(f"Error loading image {image_path}: {e}")
        return None

#path = r"C:\Users\gusta\OneDrive\Escritorio\Capstone\Crops Data\Cashew_only"
path = r"C:\Users\gusta\OneDrive\Escritorio\Capstone\transformed_image"
# Load images with error handling
def load_dataset_with_error_handling(path, batch_size=32, seed=123, validation_split=0.2, subset='both'):
    train_ds, test_ds = keras.utils.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
        subset=subset
    )
    
    # Check if images are corrupted during loading
    corrupted_images = []
    for images, labels in train_ds.concatenate(test_ds):
        for image, image_path in zip(images, train_ds.file_paths + test_ds.file_paths):
            if image is None:
                corrupted_images.append(image_path)
    
    # Print list of corrupted images
    if corrupted_images:
        print("Corrupted images:")
        for corrupted_image in corrupted_images:
            print(corrupted_image)

    return train_ds, test_ds

# Load dataset with error handling
train_ds, test_ds = load_dataset_with_error_handling(path)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Rescaling(scale=1/255, input_shape=(224, 224, 3)),
    keras.layers.Resizing(224, 224),

    keras.layers.Conv2D(32, (3, 3), activation='relu'),     #Layer 1
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),     #Layer 2
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),     #Layer 3
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),     #Layer 4
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),    #Layer 5
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),              #Layer 6
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(38, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Define the checkpoint callback
checkpoint_filepath = r"C:\Users\gusta\OneDrive\Escritorio\Capstone\Model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_freq='epoch',
    period=1
)

# Train the model with the checkpoint callback
history = model.fit(train_ds, epochs=20, callbacks=[checkpoint_callback])


