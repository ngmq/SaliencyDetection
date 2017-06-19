import matplotlib.pyplot as plt
import random
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
from keras import regularizers
from scipy.misc import imshow, imrotate, imsave

np.random.seed(233)

input_size = (96, 96, 3)
output_size = (48, 48)
nb_output = output_size[0] * output_size[1]
batch_size = 32
epochs = 40

print(nb_output)

def data_augmentation(x, y):
    nx = len(x)
    for i in range(nx):
        a = x[i]
        b = y[i].reshape(48, 48)
        
        a2 = np.fliplr(a)
        b2 = np.fliplr(b)
        x.append(a2)
        y.append(b2.reshape(48*48))
        
        a2 = np.flipud(a)
        b2 = np.flipud(b)
        x.append(a2)
        y.append(b2.reshape(48*48))
        
        a2 = np.rot90(a, 1)
        b2 = np.rot90(b, 1)
        x.append(a2)
        y.append(b2.reshape(48*48))
        
        a2 = np.rot90(a, 2)
        b2 = np.rot90(b, 2)
        x.append(a2)
        y.append(b2.reshape(48*48))
        
        a2 = np.rot90(a, 3)
        b2 = np.rot90(b, 3)
        x.append(a2)
        y.append(b2.reshape(48*48))
        
# x = np.array([ [[1., 2.], [3., 3.]], [[4., 5.], [31., 40.]] ])
# x = list(x)
# y = list([ [[-1., -2.], [-3., -3.]], [[-4., -5.], [-31., -40.]] ])
# print(x)
# print(y)

# data_augmentation(x, y)

# print('after:', x)
# print('size after:', len(x))

def max_squared_error(y_true, y_pred):
    return K.max(K.square(y_pred - y_true), axis=-1)
        

x_train = np.load('X_train.npy')
nx = len(x_train)
x_train = x_train.reshape(nx, 96, 96, 3)
y_train = np.load('y_train.npy')

x_train = list(x_train)
y_train = list(y_train)
nold = len(y_train)
data_augmentation(x_train, y_train)
assert len(y_train) == nold * 6
x_train = np.array(x_train)
y_train = np.array(y_train)

print(type(y_train[10]))
print(type(y_train[1000]))
print(type(y_train))

x_validation = np.load('X_validation.npy')
nx = len(x_validation)
x_validation = x_validation.reshape(nx, 96, 96, 3)
y_validation = np.load('y_validation.npy')

# x_test = np.load('X_test.npy')
# nx = len(x_test)
# x_test = x_test.reshape(nx, 96, 96, 3)
# y_test = np.load('y_test.npy')

# y_values = set(y_test.flatten())
# assert len(y_values) == 2

y_values = set(y_train.flatten())
assert len(y_values) == 2

model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=input_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
model.add(Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(nb_output, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dropout(0.5))
model.add(Dense(nb_output, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)

## binary_crossentropy loss
# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])

## L2 loss
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['binary_accuracy'])

## L0 loss
# model.compile(optimizer=opt, loss=max_squared_error, metrics=['binary_accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_validation, y_validation))

plt.plot(range(epochs), history.history['loss'], 'g-', label='Training loss')
plt.plot(range(epochs), history.history['val_loss'], 'r-', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.clf()

plt.plot(range(epochs), history.history['binary_accuracy'], 'g-', label='Training accuracy')
plt.plot(range(epochs), history.history['val_binary_accuracy'], 'r-', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')

model.save('model.h5')
print("Model has been saved.")