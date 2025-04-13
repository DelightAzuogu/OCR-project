import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Configure GPU memory growth (optional)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# === LOAD AND ENCODE DATA ===
csv_path = "dataset.csv"
df = pd.read_csv(csv_path)

# print("Sample from CSV:")
# print(df.head())

# Map characters to numbers
unique_chars = sorted(df['label'].unique())
char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
num_to_char = {idx: char for char, idx in char_to_num.items()}  # for decoding

# Add encoded label column
df['label_encoded'] = df['label'].map(char_to_num)

# print("\nEncoded labels:")
# print(df[['label', 'label_encoded']].head())

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

# === SPLIT DATA INTO TRAIN/VALIDATION ===
image_paths = df['image'].tolist()
labels = df['label_encoded'].tolist()

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# === CREATE DATASETS ===
BATCH_SIZE = 20

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(process_image).shuffle(1000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(process_image).batch(BATCH_SIZE)

# === DEFINE CNN MODEL ===
num_classes = len(char_to_num)

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Show model summary
model.summary()

# === COMPILE MODEL ===
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === TRAIN MODEL ===
EPOCHS = 10
BATCH_SIZE = 20

model.fit(
    train_dataset,
    # validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

model.evaluate(val_dataset, batch_size=BATCH_SIZE, verbose=2)

print(model.summary())