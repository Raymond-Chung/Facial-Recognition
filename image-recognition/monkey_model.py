import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
from keras import layers

# Add these lines to verify device setup
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Is built with CUDA? ", tf.test.is_built_with_cuda())
print("TensorFlow Version:", tf.__version__)
# ... rest of your code

img_h, img_w = 180, 180
batch = 20

test_ds = tf.keras.utils.image_dataset_from_directory(
    "../images/monkeys",
    validation_split = 0.1,
    subset = "validation",
    seed = 123,
    image_size = (img_h, img_w),
    batch_size = batch
)

train_ds = tf.keras.utils.image_dataset_from_directory(
    "../images/monkeys",
    validation_split = 0.11111112,
    subset = "training",
    seed = 123,
    image_size = (img_h, img_w),
    batch_size = batch
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "../images/monkeys",
    validation_split = 0.11111112,
    subset = "validation",
    seed = 123,
    image_size = (img_h, img_w),
    batch_size = batch
)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):     # take 1 = take a single batch
    for i in range(9):
        ax = plt.subplot(3, 3, i +1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()

num_classes = len(class_names)
model = tf.keras.Sequential(
    [
    layers.Rescaling(1./255, input_shape=(180,180, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'), 
	layers.MaxPooling2D(), 
	layers.Conv2D(32, 3, padding='same', activation='relu'), 
	layers.MaxPooling2D(), 
	layers.Conv2D(64, 3, padding='same', activation='relu'), 
	layers.MaxPooling2D(), 
	layers.Flatten(), 
	layers.Dense(128, activation='relu'), 
    layers.Dense(num_classes)
    ]
)

model.compile(
    optimizer = "adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3
)

model.evaluate(test_ds)

# shuffled_ds = test_ds.unbatch().shuffle(1000)

plt.figure(figsize=(10, 10))

shuffled_ds = test_ds.unbatch().shuffle(1000).batch(9)
for images, labels in shuffled_ds.take(1):
    classifications = model(images)
    print(classifications) 

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index = numpy.argmax(classifications[i])
        plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])

plt.show()

