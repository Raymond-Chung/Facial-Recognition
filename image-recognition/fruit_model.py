"""
model for fruits 
youtube turtorial - https://www.youtube.com/watch?v=ba42uYJd8nc&t=615s
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

img_h, img_w = 32, 32
batch = 20

# Found 460 files belonging to 3 classes.
# Total images in train folder
train_ds = tf.keras.utils.image_dataset_from_directory(
    "../images/fruits/train",
    image_size = (img_h, img_w),
    batch_size = batch
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "../images/fruits/validation",
    image_size = (img_h, img_w),
    batch_size = batch
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "../images/fruits/test",
    image_size = (img_h, img_w),
    batch_size = batch
)

# display the dataset
class_names = ["apple", "banana", "oragne"]
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):     # take 1 = take a single batch
    for i in range(9):
        ax = plt.subplot(3, 3, i +1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()

# neural network
# similar to human brain
# design to recognize patterns and make predictions from data
# convolutional neural network (CNN) = proccess images through mult. layers to see patterns

model = tf.keras.Sequential(
    [
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(3)
    ]
)

model.compile(
    optimizer = "adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)

# train data
# loss indicates how well model is predicting, higher values = poorly, lower = better
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10 #iterations of data set)
)

# eval on unseen data
model.evaluate(test_ds)

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    classifications = model(images)
    # print(classifications) # prints 3 numbers per row, highest val in row = what model thinks. order is apple, banana, orange

    # display images in 3x3 with predictions and actual
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index = numpy.argmax(classifications[i])
        plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])

plt.show()
