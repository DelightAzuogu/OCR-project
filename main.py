import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# === Step 1: Load CSV ===
csv_path = "dataset.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# Show the first few rows to check
print("Sample from CSV:")
print(df.head())

# === Step 2: Encode Labels ===
# Get all unique labels (e.g., A, b, 7, etc.)
unique_chars = sorted(df['label'].unique())
char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
num_to_char = {idx: char for char, idx in char_to_num.items()}  # for decoding later

# Add a new column with numeric labels
df['label_encoded'] = df['label'].map(char_to_num)

print("\nEncoded labels:")
print(df[['label', 'label_encoded']].head())

# === Step 3: Define image processing function ===
IMG_HEIGHT = 64
IMG_WIDTH = 64

def process_image(file_path, label):
    # Convert file path and label to tensors
    file_path = tf.convert_to_tensor(file_path)
    label = tf.convert_to_tensor(label)

    # Read and decode image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)  # 1 = grayscale

    # Resize and normalize
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0

    return image, label

# === Step 4: Prepare the tf.data.Dataset ===
image_paths = df['image'].tolist()
labels = df['label_encoded'].tolist()

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(process_image)

# Shuffle and batch
BATCH_SIZE = 20
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

# === Optional: Show one batch to verify ===
for images, labels in dataset.take(50):
    print(f"\nBatch shape: images={images.shape}, labels={labels.shape}")
    print("First label (numeric):", labels[0].numpy())
    print("First label (char):", num_to_char[labels[0].numpy()])
