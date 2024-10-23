import numpy as np
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score
import keras_tuner as kt
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set environment variable to reduce TensorFlow log verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR messages

# Load images and handle errors
def load_dataset_with_error_handling(path, batch_size=32, seed=123, validation_split=0.2, subset='both'):
    train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
        subset=subset
    )
    return train_ds, test_ds

# Enable GPU usage and restrict memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs detected: ", len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs detected. Training will use CPU.")

# Define the model building function for Keras Tuner
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=1/255, input_shape=(224, 224, 3)),
        tf.keras.Sequential([
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.2),
        ], name='augmentation'),
        tf.keras.layers.Conv2D(
            hp.Int('conv_1_units', min_value=64, max_value=256, step=64), 
            (3, 3), 
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(
            hp.Int('conv_2_units', min_value=64, max_value=256, step=64), 
            (3, 3), 
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Conv2D(
            hp.Int('conv_3_units', min_value=128, max_value=512, step=128), 
            (3, 3), 
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            hp.Int('dense_units', min_value=64, max_value=256, step=64), 
            activation='relu'),
        tf.keras.layers.Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(36, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Define the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    directory='kt_tuner_dir',
    project_name='image_classification_tuning'
)

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Load dataset
path = r"C:\Users\gustavo\Desktop\Capstone\Crops Data\Cashew_only_inval - Copy"
train_ds, test_ds = load_dataset_with_error_handling(path)

# Start hyperparameter tuning
tuner.search(train_ds, validation_data=test_ds, epochs=50, callbacks=[stop_early])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Summary of the best model
model.summary()

# Define additional callbacks
checkpoint_filepath = r"C:\Users\gustavo\Desktop\Capstone\Model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_freq='epoch'
)

# Measure training time
start_time = time.time()

# Train the model with the best hyperparameters
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=250,
    callbacks=[checkpoint_callback, stop_early]
)

end_time = time.time()
training_duration = end_time - start_time
print(f"Training completed in {training_duration:.2f} seconds")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the training history
plot_history(history)

# Evaluate model and print precision and F1 score
def evaluate_model(model, test_ds):
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred = model.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)

    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

evaluate_model(model, test_ds)

# Create confusion matrix and plot
def plot_confusion_matrix(model, test_ds):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=test_ds.class_names, yticklabels=test_ds.class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix with Percentages')
    plt.show()

    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=test_ds.class_names))

plot_confusion_matrix(model, test_ds)
