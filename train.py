import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os, shutil
from tensorflow.keras.utils import get_file


DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    print("Downloading sample dataset (Cats vs Dogs)...")
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    path = get_file("cats_and_dogs_filtered.zip", origin=url, extract=True)
    base = os.path.join(os.path.dirname(path), "cats_and_dogs_filtered")
    train_src = os.path.join(base, "train")
    val_src = os.path.join(base, "validation")
    os.makedirs(DATA_DIR, exist_ok=True)
    shutil.copytree(train_src, os.path.join(DATA_DIR, "train"), dirs_exist_ok=True)
    shutil.copytree(val_src, os.path.join(DATA_DIR, "val"), dirs_exist_ok=True)
    print("Dataset ready under data/train and data/val")
else:
    print("Dataset folder already exists, skipping download.")


IMG_SIZE = (128,128)
BATCH = 16
EPOCHS = 8

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")


train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical'
)

val_flow = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical'
)


model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_flow.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n Model Summary:")
model.summary()


print("\n Starting training...\n")
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    verbose=1
)


os.makedirs("model", exist_ok=True)
model.save("model/cnn_model.h5")
print("\n Model saved to model/cnn_model.h5")
print("Class indices:", train_flow.class_indices)


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o', color='red')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o', color='orange')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("model/training_plot.png")
plt.show()

print("\n Training visualization saved as model/training_plot.png")
