from tensorflow.keras import layers
from tensorflow.keras.models import Model
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten


def cnn_model(size):
    input = layers.Input(shape=(size, size, 1))

    conv1 = layers.Conv2D(128, (2, 2), activation='relu', padding='same')(input) 
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    dropout1 = layers.Dropout(0.25)(pool1)

    conv2 = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(dropout1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout2 = layers.Dropout(0.25)(pool2)

    flatten1 = layers.Flatten()(dropout2)
#     dense1 = layers.Dense(24000, activation='relu')(flatten1)
    dropout3 = layers.Dropout(0.25)(flatten1)
    dense2 = layers.Dense(10000, activation='softmax')(dropout3)

    model = Model(input, dense2)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    print(model.summary())
    return model
