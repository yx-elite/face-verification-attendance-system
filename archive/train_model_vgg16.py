import keras, os
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam


IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 64

train_path = 'data/anchor'
valid_path = 'data/positive'

vgg16_conv = VGG16(
    include_top=False, 
    input_shape=IMAGE_SIZE, 
    weights='imagenet'
)

for layer in vgg16_conv.layers:
    layer.trainable = False

# Data preprocessing & Data augmentation
train_data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_data_gen = ImageDataGenerator(
    rescale=1./255
)

train_dataset = train_data_gen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
)

n_classes = len(train_dataset.class_indices)

# Flatten VGG16 output (fully-connected layers) into 1-D array and serves as input for Dense layer
layer_top = vgg16_conv.output
layer_top = Flatten()(layer_top)
layer_out_pred = Dense(units=n_classes, activation='softmax')(layer_top)

model = Model(inputs=vgg16_conv.inputs, outputs=layer_out_pred)
model.summary()

# Compile the model for training
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

test_dataset = test_data_gen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
)

vgg16_train = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    steps_per_epoch=len(train_dataset),
    validation_steps=len(test_dataset)
)

# Plot loss and accuracy of model trained
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

axs[0].plot(vgg16_train.history['loss'], label='train loss')
axs[0].plot(vgg16_train.history['val_loss'], label='val loss')
axs[0].legend()
axs[0].set_title('Loss and Validation Loss')

axs[1].plot(vgg16_train.history['accuracy'], label='train acc')
axs[1].plot(vgg16_train.history['val_accuracy'], label='val acc')
axs[1].legend()
axs[1].set_title('Accuracy and Validation Accuracy')

plt.savefig('LossAcc_subplot')
plt.show()

model.save('face_recognition_model.h5')