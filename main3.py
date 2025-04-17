import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

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

# Image size config
IMG_HEIGHT = 64
IMG_WIDTH = 64

# === IMAGE PROCESSING FUNCTION ===
def process_image(file_path, label):
    file_path = tf.convert_to_tensor(file_path)
    label = tf.convert_to_tensor(label)

    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)  # grayscale
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # normalize to [0,1]

    return image, label

# === PREPARE TRAINING DATA ===
train_image_paths = train_df['image'].tolist()
train_labels = train_df['label_encoded'].tolist()

# === PREPARE TEST DATA ===
test_image_paths = test_df['image'].tolist()
test_labels = test_df['label_encoded'].tolist()

# === CREATE DATASETS ===
BATCH_SIZE = 20

# Training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(process_image).shuffle(1000).batch(BATCH_SIZE)

# Test dataset (from dataset.csv)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(process_image).batch(BATCH_SIZE)

# === DEFINE CNN MODEL ===
num_classes = len(char_to_num)

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

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
# Note: Without validation data, we can't use validation-based callbacks
# Using only learning rate scheduler
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='accuracy',
    save_best_only=True,
    mode='max',
    verbose=0
)

# === TRAIN MODEL ===
EPOCHS = 50

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[lr_callback, checkpoint],
    verbose=1
)

# === EVALUATE MODEL ON TEST SET ===
print("\nEvaluating model on test dataset:")
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Optional: Plot training history
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
except:
    pass