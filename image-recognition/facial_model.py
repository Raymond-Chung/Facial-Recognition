import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
from keras import layers

img_h, img_w = 90, 90
batch = 64

train_ds = tf.keras.utils.image_dataset_from_directory (
    "/Users/raymondchung/Documents/CS/project/images/facial_emotions/train",
    image_size = (img_h, img_w),
    batch_size = batch
)

test_ds = tf.keras.utils.image_dataset_from_directory (
    "/Users/raymondchung/Documents/CS/project/images/facial_emotions/test",
    image_size = (img_h, img_w),
    batch_size = batch
)

class_names = train_ds.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):     # take 1 = take a single batch
#     for i in range(9):
#         ax = plt.subplot(3, 3, i +1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

num_classes = len(class_names)
model = tf.keras.Sequential(
    [  
    # tf.keras.Input(shape=(90, 90, 3)),
    layers.Rescaling(1./255),
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

model_info = model.fit(
    train_ds,
    validation_data = test_ds,
    epochs = 10 #iterations of data set
)

model.evaluate(test_ds)

# display data
accuracy = model_info.history['accuracy']
val_accuracy = model_info.history['val_accuracy']
loss = model_info.history['loss']
val_loss = model_info.history['val_loss']

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='accuracy')
plt.plot(val_accuracy, label='val accuracy')
plt.title('Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# loss graph
plt.subplot(1, 2, 2)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val loss')
plt.title('Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# plt.show()

file_name = "emotion_model.keras"
model.save(file_name)
emotion_dict = {}
count = 0
for i in range(len(class_names)):
    emotion_dict[i] = class_names[i]
# print(emotion_dict)
