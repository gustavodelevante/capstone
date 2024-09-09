import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define a custom callback to stop training once accuracy reaches 94%
class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            if logs.get('val_accuracy') >= 0.90:        #change to 90
                print("\nReached 94% accuracy so cancelling training!")
                self.model.stop_training = True

# Load images and handle errors
def load_dataset_with_error_handling(path, batch_size=128, seed=123, validation_split=0.2, subset='both'):
    train_ds, test_ds = keras.utils.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
        subset=subset
    )
    
    corrupted_images = []
    for images, labels in train_ds.concatenate(test_ds):
        for image, image_path in zip(images, train_ds.file_paths + test_ds.file_paths):
            if image is None:
                corrupted_images.append(image_path)
    
    if corrupted_images:
        print("Corrupted images:")
        for corrupted_image in corrupted_images:
            print(corrupted_image)

    return train_ds, test_ds

path = r"C:\Users\gustavo\Desktop\Capstone\Crops Data\Cashew_only_inval"
train_ds, test_ds = load_dataset_with_error_handling(path)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Rescaling(scale=1/255, input_shape=(224, 224, 3)),
   

    tf.keras.Sequential([
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2),
    ], name='augmentation'),


    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
   

    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.4),

    
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.5),  
    
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Dropout(0.6),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(21, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

checkpoint_filepath = r"C:\Users\gustavo\Desktop\Capstone\Model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_freq='epoch'
)

accuracy_threshold_callback = AccuracyThresholdCallback()

# Define EarlyStopping to halt training if accuracy decreases
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,     #change to 10
    mode='auto',  # 'auto' mode means it infers the direction based on the monitored quantity
    restore_best_weights=True
)

# Train the model with the callbacks
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=35,
    callbacks=[checkpoint_callback, accuracy_threshold_callback, early_stopping_callback]
)
