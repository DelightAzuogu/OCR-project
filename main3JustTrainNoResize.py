import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # Removed ReduceLROnPlateau

# Configure GPU memory growth (optional)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# === LOAD AND ENCODE DATA ===
train_csv_path = "train.csv"
test_csv_path = "dataset.csv"

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Combine unique characters from both datasets to ensure consistent encoding
all_chars = set(train_df['label'].unique()).union(set(test_df['label'].unique()))
unique_chars = sorted(list(all_chars))

char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
num_to_char = {idx: char for char, idx in char_to_num.items()}  # for decoding

# Add encoded label column to both dataframes
train_df['label_encoded'] = train_df['label'].map(char_to_num)
test_df['label_encoded'] = test_df['label'].map(char_to_num)

# Image size config - using known dimensions
IMG_HEIGHT = 40
IMG_WIDTH = 60

# === IMAGE PROCESSING FUNCTION ===
def process_image(file_path, label):
    # Using label parameter even though it's not directly processed
    file_path = tf.convert_to_tensor(file_path)
    label = tf.convert_to_tensor(label)

    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)  # grayscale

    # Convert to float32 before division
    image = tf.cast(image, tf.float32)
    image = image / 255.0  # normalize to [0,1]

    return image, label

# === PREPARE TRAINING DATA ===
train_image_paths = train_df['image'].tolist()
train_labels = train_df['label_encoded'].tolist()

# === CREATE DATASETS ===
BATCH_SIZE = 20

# Training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for better performance

# === DEFINE CNN MODEL ===
num_classes = len(char_to_num)

# Define model with fixed input shape
model = models.Sequential([
    # First layer with fixed input shape
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.005),  # Reduced regularization
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.005)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.005)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.005)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Flatten(),  # We can use Flatten since dimensions are fixed
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Print model summary
model.summary()

# === DEFINE LEARNING RATE AND OPTIMIZER ===
initial_learning_rate = 0.001

def lr_schedule(epoch):
    return initial_learning_rate * (0.9 ** (epoch // 10))

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# === COMPILE MODEL ===
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === CALLBACKS ===
checkpoint = ModelCheckpoint(
    'best_model1.h5',
    monitor='accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# === TRAIN MODEL ===
EPOCHS = 50

# Initialize history variable
history = None

try:
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[lr_callback, checkpoint, early_stopping],
        verbose=1
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")

    # Try with smaller batch size if there was an error
    try:
        print("Retrying with smaller batch size...")
        BATCH_SIZE = 10
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
        train_dataset = train_dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            callbacks=[lr_callback, checkpoint, early_stopping],
            verbose=1
        )
        print("Training completed successfully with smaller batch size!")
    except Exception as e2:
        print(f"Error during retry: {e2}")

# Optional: Plot training history
if history is not None:
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history saved to 'training_history.png'")
    except Exception as e:
        print(f"Error creating plot: {e}")
else:
    print("No training history available to plot.")