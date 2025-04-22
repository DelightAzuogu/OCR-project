import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs

import pandas as pd
import tensorflow as tf
import json
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

# Configure GPU memory growth (optional)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# === LOAD AND ENCODE DATA ===
train_csv_path = "train.csv"
train_df = pd.read_csv(train_csv_path)

# Get unique characters for encoding
unique_chars = sorted(list(set(train_df['label'].unique())))

char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Save character mappings to JSON file
with open('char_mappings.json', 'w') as f:
    json.dump({
        'char_to_num': char_to_num,
        'num_to_char': num_to_char
    }, f, indent=4)

# Add encoded label column to dataframe
train_df['label_encoded'] = train_df['label'].map(char_to_num)

# Image size config
IMG_HEIGHT = 60
IMG_WIDTH = 40

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

# === CREATE DATASET ===
BATCH_SIZE = 20

# Training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(process_image).shuffle(1000).batch(BATCH_SIZE)

# === DEFINE CNN MODEL ===
num_classes = len(char_to_num)

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    # First block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Second block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Third block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Dense layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# === DEFINE LEARNING RATE AND OPTIMIZER ===
initial_learning_rate = 0.001

def lr_schedule(epoch):
    return initial_learning_rate * (0.9 ** (epoch // 10))

lr_callback = LearningRateScheduler(lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# === COMPILE MODEL ===
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === CALLBACKS ===
checkpoint = ModelCheckpoint(
    'best_model11.h5',  # Still save h5 during training for checkpointing
    monitor='accuracy',
    save_best_only=True,
    mode='max',
    verbose=0
)

# === TRAIN MODEL ===
EPOCHS = 50

print("Training model...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[lr_callback, checkpoint],
    verbose=1
)

# === SAVE MODEL AS PB FILE ===
# First load the best model from checkpoint
model.load_weights('best_model11.h5')

# Save in TensorFlow SavedModel format (.pb)
export_path = 'saved_model11'
print(f"Exporting trained model to {export_path}")
tf.saved_model.save(model, export_path)
print(f"Model exported to: {export_path}")

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