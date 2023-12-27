import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 64  # Adjust batch size
EPOCHS = 100  # Increase the number of epochs
PATIENCE = 10  # Number of epochs with no improvement after which training will be stopped

train_path = 'data/anchor'
valid_path = 'data/positive'

# Data preprocessing & Data augmentation
train_data_gen = ImageDataGenerator(
    rotation_range=20,  # Adjust data augmentation parameters
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
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

# Load pre-trained VGG16 model
vgg16_conv = VGG16(
    include_top=False,
    input_shape=IMAGE_SIZE,
    weights='imagenet'
)

# Fine-tune the last few layers of VGG16
for layer in vgg16_conv.layers[:-2]:
    layer.trainable = False

# Flatten VGG16 output (fully-connected layers) into 1-D array and serves as input for Dense layer
layer_top = vgg16_conv.output
layer_top = Flatten()(layer_top)
layer_out_pred = Dense(units=n_classes, activation='softmax')(layer_top)

model = Model(inputs=vgg16_conv.inputs, outputs=layer_out_pred)
model.summary()

# Compile the model for training with a reduced learning rate
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

# Evaluate the model on the validation set
test_dataset = test_data_gen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
)

# Train the model
vgg16_train = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=len(train_dataset),
    validation_data=test_dataset,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('model/new_model1.h5')

# Plot training history (loss and accuracy)
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(vgg16_train.history['accuracy'])
plt.plot(vgg16_train.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(vgg16_train.history['loss'])
plt.plot(vgg16_train.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('LossAcc_subplot')

# Generate predictions on the validation set
predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_dataset.classes
class_labels = list(test_dataset.class_indices.keys())

# Generate and print the classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Generate and plot the confusion matrix
confusion_mtx = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)

thresh = confusion_mtx.max() / 2.
for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
    plt.text(j, i, format(confusion_mtx[i, j], 'd'), horizontalalignment="center", color="white" if confusion_mtx[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('Confusion Matrix')

# Display overall model accuracy
vgg_acc = accuracy_score(true_classes, predicted_classes)
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc * 100))