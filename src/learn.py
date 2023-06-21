import pandas as pd
from tqdm import tqdm
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow_addons as tfa

IMAGE_DIR = "proc_data/pieces/square/"
FRAME_SIZE = 100
FRAME_DEPTH = 3

df = pd.read_csv("proc_data/final_data.csv")

CLASSES = len(df.columns) - 1

print(df.head())
print(df.columns)
print(len(df.index))

x_data = []
for i in tqdm(range(df.shape[0])):
    img = load_img(IMAGE_DIR + df['image_name'][i], target_size = (FRAME_SIZE, FRAME_SIZE, FRAME_DEPTH))
    img = img_to_array(img)
    img = img / 255.
    x_data.append(img)

x = np.array(x_data)
print(x)

y = np.array(df.drop(columns = ['image_name']))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(FRAME_SIZE, FRAME_SIZE, FRAME_DEPTH)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(CLASSES, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=CLASSES)])

history = model.fit(x_train, y_train, epochs=60, batch_size=64, validation_data=(x_test, y_test)) # GET VALIDATION DATA

model.save("models/base_model")

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

acc = history.history['precision']
val_acc = history.history['val_precision']
plt.plot(epochs, acc, 'y', label='Training precision')
plt.plot(epochs, val_acc, 'r', label='Validation precision')
plt.title('Training and validation precision')
plt.xlabel('Epochs')
plt.ylabel('Precisoin')
plt.legend()
plt.show()

acc = history.history['recall']
val_acc = history.history['val_recall']
plt.plot(epochs, acc, 'y', label='Training recall')
plt.plot(epochs, val_acc, 'r', label='Validation recall')
plt.title('Training and validation recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.show()

acc = history.history['f1_score']
val_acc = history.history['val_f1_score']
plt.plot(epochs, acc, 'y', label='Training F1')
plt.plot(epochs, val_acc, 'r', label='Validation F1')
plt.title('Training and validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

acc = model.evaluate(x_test, y_test)
print("Accuracy = ", acc)