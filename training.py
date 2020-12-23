from preprocess import *
from time import time
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras import regularizers

X_train, X_test, y_train, y_test = get_train_test()
X_train = X_train.reshape(X_train.shape[0], 20, 25, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 25, 1)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

## Creating the convolutional base
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 25, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.50))
model.add(Dense(2, activation='softmax')) ###
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, y_train_hot, batch_size=100, epochs=200, verbose=1, validation_data=(X_test, y_test_hot), callbacks=[tensorboard])

# evaluate the model
scores = model.evaluate(X_train, y_train_hot, verbose=0)
print("train_set")
print(model.metrics_names)
print(scores)

scores = model.evaluate(X_test, y_test_hot, verbose=0)
print("test_set")
print(model.metrics_names)
print(scores)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

'''
# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
'''
'''
#save model using tensorflow method
saver = tf.train.Saver()
sess = keras.backend.get_session()
saver.save(sess, './keras_model')

model.save('keras_model.hdf5')
'''
print("Saved model to disk")
print(model.summary())