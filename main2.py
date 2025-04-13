import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

csv_path = "dataset.csv"
df = pd.read_csv(csv_path)

print("Sample from CSV:")
print(df.head())

unique_chars = sorted(df['label'].unique())
char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
num_to_char = {idx: char for char, idx in char_to_num.items()}  # This is used for decoding the numbers

df['label_encoded'] = df['label'].map(char_to_num)

IMG_HEIGHT = 128
IMG_WIDTH = 128

# Split the data into training and test sets (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_encoded'])

def process_image(file_path, label):
    file_path = tf.convert_to_tensor(file_path)
    label = tf.convert_to_tensor(label)

    image = tf.io.read_file(file_path)
    # grey scale the image
    image = tf.image.decode_png(image, channels=1)

    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0

    return image, label

# Create training dataset
train_image_paths = train_df['image'].tolist()
train_labels = train_df['label_encoded'].tolist()
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(process_image)

# Create test dataset
test_image_paths = test_df['image'].tolist()
test_labels = test_df['label_encoded'].tolist()
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(process_image)

BATCH_SIZE = 20
# Apply batching and shuffling to training data
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)
# Only batch the test data (no need to shuffle)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Build the CNN model
num_classes = len(unique_chars)


model = models.Sequential([
    # First convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Second convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Third convolutional layer
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Flatten the output and feed it into dense layers
    layers.Flatten(),
    layers.Dropout(0.5),  # Add dropout to reduce overfitting
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Output layer
])
# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
EPOCHS = 10
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_dataset)
print(f'\nTest accuracy: {test_acc:.4f}')