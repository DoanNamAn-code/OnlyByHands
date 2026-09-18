import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
train_data, test_data = tfds.load('emnist/balanced', split=['train', 'test'], as_supervised=True, batch_size=-1)
x_train, y_train = tfds.as_numpy(train_data)
x_test, y_test = tfds.as_numpy(test_data)
x_train = np.transpose(x_train, (0, 2, 1, 3)).astype('float32') / 255.0
x_test = np.transpose(x_test, (0, 2, 1, 3)).astype('float32') / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(47, activation='softmax') 
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))
model.save('emnist_model.h5')
