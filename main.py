import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

csv_path = "dataset.csv" 
df = pd.read_csv(csv_path)

print("Sample from CSV:")
print(df.head())

unique_chars = sorted(df['label'].unique())
char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
num_to_char = {idx: char for char, idx in char_to_num.items()}  # This is used for decoding the numbers

df['label_encoded'] = df['label'].map(char_to_num)

print("\nEncoded labels:")
print(df[['label', 'label_encoded']].head())

IMG_HEIGHT = 64
IMG_WIDTH = 64

def process_image(file_path, label):
    file_path = tf.convert_to_tensor(file_path)
    label = tf.convert_to_tensor(label)

    image = tf.io.read_file(file_path)
    # grey scale the image
    image = tf.image.decode_png(image, channels=1) 

    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0

    return image, label

image_paths = df['image'].tolist()
labels = df['label_encoded'].tolist()

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(process_image)

BATCH_SIZE = 20
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

# Show one batch to verify 
for images, labels in dataset.take(50):
    print(f"\nBatch shape: images={images.shape}, labels={labels.shape}")
    print("First label (numeric):", labels[0].numpy())
    print("First label (char):", num_to_char[labels[0].numpy()])
