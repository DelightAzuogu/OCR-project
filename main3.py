import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt

# Enable memory growth for GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load and prepare dataset
csv_path = "dataset.csv"
df = pd.read_csv(csv_path)

# Check class distribution to identify imbalance
class_distribution = df['label'].value_counts()
min_samples = class_distribution.min()
max_samples = class_distribution.max()

unique_chars = sorted(df['label'].unique())
char_to_num = {char: idx for idx, char in enumerate(unique_chars)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

df['label_encoded'] = df['label'].map(char_to_num)

# Image dimensions - consider increasing resolution
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Split with stratification to maintain class distribution
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_encoded'])

# Data augmentation function for training
def augment_image(image):
    # Random rotation
    image = tf.image.random_rotation(image, 0.1)
    # Random zoom (scale)
    image = tf.image.resize_with_crop_or_pad(
        image, int(IMG_HEIGHT * 1.2), int(IMG_WIDTH * 1.2))
    image = tf.image.central_crop(image, np.random.uniform(0.8, 1.0))
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # Random brightness and contrast
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    # Ensure values stay in [0, 1]
    image = tf.clip_by_value(image, 0, 1)
    return image

def process_image_train(file_path, label):
    file_path = tf.convert_to_tensor(file_path)
    label = tf.convert_to_tensor(label)

    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0
    # Apply augmentation only to training data
    image = augment_image(image)

    return image, label

def process_image_test(file_path, label):
    file_path = tf.convert_to_tensor(file_path)
    label = tf.convert_to_tensor(label)

    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0

    return image, label

# Create training dataset with augmentation
train_image_paths = train_df['image'].tolist()
train_labels = train_df['label_encoded'].tolist()
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(process_image_train)

# Create test dataset without augmentation
test_image_paths = test_df['image'].tolist()
test_labels = test_df['label_encoded'].tolist()
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(process_image_test)

# Split the test set into validation and test
test_dataset_size = len(test_labels)
val_size = test_dataset_size // 2
val_dataset = test_dataset.take(val_size)
test_dataset = test_dataset.skip(val_size)

# Optimize batch size based on dataset size and GPU memory
BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(buffer_size=5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Display a few augmented samples
plt.figure(figsize=(12, 8))
for images, labels in train_dataset.take(1):
    for i in range(min(9, BATCH_SIZE)):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Label: {num_to_char[labels[i].numpy()]}')
        plt.axis('off')
plt.savefig('augmented_samples.png')
plt.close()

# Build improved CNN model
num_classes = len(unique_chars)

def create_model():
    # Using a modified architecture inspired by popular OCR models
    model = models.Sequential([
        # First block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Second block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Third block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Final layers
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

model = create_model()

# Use a learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

# Compile with adaptive learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add callbacks for better training
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    ),
    callbacks.ModelCheckpoint(
        filepath='best_ocr_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Train with validation data
EPOCHS = 150
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks_list
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.close()

# Load the best model
best_model = tf.keras.models.load_model('best_ocr_model.h5')

# Evaluate on test data
test_loss, test_acc = best_model.evaluate(test_dataset)
print(f'\nTest accuracy: {test_acc:.4f}')

# Analyze predictions on test data
def analyze_predictions():
    predictions = []
    true_labels = []
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for images, labels in test_dataset:
        preds = best_model.predict(images)
        pred_classes = np.argmax(preds, axis=1)

        for i in range(len(labels)):
            true_class = labels[i].numpy()
            pred_class = pred_classes[i]
            confusion_matrix[true_class, pred_class] += 1

            if true_class != pred_class:
                predictions.append(pred_class)
                true_labels.append(true_class)

    # Identify the most confused classes
    class_errors = []
    for i in range(num_classes):
        total = np.sum(confusion_matrix[i, :])
        correct = confusion_matrix[i, i]
        error_rate = 1.0 - (correct / total) if total > 0 else 0
        class_errors.append((i, error_rate))

    class_errors.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 most confused classes:")
    for class_idx, error_rate in class_errors[:5]:
        print(f"Class {num_to_char[class_idx]}: {error_rate:.2%} error rate")

    return confusion_matrix

confusion_matrix = analyze_predictions()

# Sample prediction visualization
plt.figure(figsize=(12, 8))
count = 0
for images, labels in test_dataset.take(1):
    preds = best_model.predict(images)
    pred_classes = np.argmax(preds, axis=1)

    for i in range(min(9, len(labels))):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        true_label = num_to_char[labels[i].numpy()]
        pred_label = num_to_char[pred_classes[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'True: {true_label}, Pred: {pred_label}', color=color)
        plt.axis('off')
        count += 1
        if count >= 9:
            break

plt.savefig('prediction_samples.png')
plt.close()