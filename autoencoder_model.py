from tensorflow.keras import layers
from tensorflow.keras.models import Model
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten


def autoencoder_model(size):
    input = layers.Input(shape=(size, size, 1))

    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input) 
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)

    # Decoder
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    up1 = layers.UpSampling2D((2,2))(conv3)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up1)

    # Autoencoder
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    print(autoencoder.summary())
    return autoencoder
