import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from baseline_utils import generate_grid_coordinates, generate_grid_values_from_sourceloc, generate_sourceloc_from_coordinates, add_anomaly
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.datasets import mnist
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tensorflow.keras.layers import Dense, Input, Flatten,Reshape, LeakyReLU as LR,Activation, Dropout
from tensorflow.keras.models import Model, Sequential
import argparse
import os
from autoencoder_model import autoencoder_model
from cnn_model import cnn_model
from mean_reconstruction import mean_reconstruction
from plot_grid import plot_grid
plt.rcParams["figure.figsize"] = (13,7)


def main():
    parser=argparse.ArgumentParser(description='Anomaly-proof localization in IoT')
    parser.add_argument('--aoi_size',type=int,default=100)
    parser.add_argument('--source_intensity',type=int,default=10000000)
    parser.add_argument('--detector_area',type=float,default=0.02)
    parser.add_argument('--efficiency',type=float,default=1.0)
    parser.add_argument('--data_samples',type=int,default=5000)
    parser.add_argument('--test_train_split',type=float,default=0.3)
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--load_pretrained', type=bool, default=True)
    
    print("Executing the program with the following arguments")
    print(parser)
    
    args = parser.parse_args()
    size = args.aoi_size
    source_intensity = args.source_intensity
    detector_area = args.detector_area
    efficiency = args.efficiency
    data_samples = args.data_samples
    test_train_split = args.test_train_split
    generate_data = args.generate_data
    load_pretrained = args.load_pretrained
    
    print(args.generate_data)
    print(args.load_pretrained)

    try:
        os.mkdir('saved_models')
    except:
        pass
    try:
        os.mkdir('saved_datasets')
    except:
        pass

    if generate_data:
        print("Generating data ", data_samples, " samples")
        source_coordinates = np.random.rand(data_samples, 2) * (size**2) ## random source locations
        np.save('saved_datasets/source_coordinates_{}x{}.npy'.format(size, size), source_coordinates)
    else:
        print("Loading saved data")
        source_coordinates = np.load('saved_datasets/source_coordinates_{}x{}.npy'.format(size, size))
    grid_coordinates = generate_grid_coordinates(size) ## generate sensor locations for training (each sensor placed at center of each grid)
    sensor_locations = grid_coordinates.reshape(size**2, 2) ## sensor locations

    if generate_data:
        x = generate_grid_values_from_sourceloc(source_coordinates, size, grid_coordinates, source_intensity, detector_area, efficiency) 
        y = generate_sourceloc_from_coordinates(source_coordinates, size)
        np.save('saved_datasets/cpm_values_{}x{}.npy'.format(size, size), x)
        np.save('saved_datasets/source_grid_coordinates_{}x{}.npy'.format(size, size), y)
        print("Data generation successful")
    else:
        x = np.load('saved_datasets/cpm_values_{}x{}.npy'.format(size, size))
        y = np.save('saved_datasets/source_grid_coordinates_{}x{}.npy'.format(size, size))
        print("Data successfully loaded")
        
        
    y = y.reshape(x.shape[0],size**2) ## input
    X, Y = shuffle(x, y) ## output

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_train_split, random_state=42)

    # Normalizing train values
    X_train_scaled = X_train.copy()
    for i in range(X_train_scaled.shape[0]):
        max_element = np.amax(X_train_scaled[i])
        min_element = np.amin(X_train_scaled[i])
        for j in range(X_train_scaled[i].shape[0]):
            for k in range(X_train_scaled[i][j].shape[0]):
                X_train_scaled[i][j][k] = (X_train_scaled[i][j][k] - min_element) / (max_element - min_element)

    # Normalizing test values
    X_test_scaled = X_test.copy()
    for i in range(X_test_scaled.shape[0]):
        max_element = np.amax(X_test_scaled[i])
        min_element = np.amin(X_test_scaled[i])
        for j in range(X_test_scaled[i].shape[0]):
            for k in range(X_test_scaled[i][j].shape[0]):
                X_test_scaled[i][j][k] = (X_test_scaled[i][j][k] - min_element) / (max_element - min_element)

    ## Adding noise to 1000 random grids
    noisy_train_data = add_anomaly(X_train.reshape(X_train.shape[0], size, size), 1000, size)
    noisy_test_data = add_anomaly(X_test.reshape(X_test.shape[0], size, size), 1000, size)
    print("Noise added to data")

    ## Reshaping and type setting
    X_train = X_train.reshape(X_train.shape[0], size, size, 1)
    X_test = X_test.reshape(X_test.shape[0], size, size, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], size, size, 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], size, size, 1)
    X_train_scaled = X_train_scaled.astype('float32')
    X_test_scaled = X_test_scaled.astype('float32')
    noisy_train_data = noisy_train_data.reshape(noisy_train_data.shape[0], size, size, 1)
    noisy_test_data = noisy_test_data.reshape(noisy_test_data.shape[0], size, size, 1)
    noisy_train_data = noisy_train_data.astype('float32')
    noisy_test_data = noisy_test_data.astype('float32')
    print("X_train.shape: ", X_train.shape)
    print("X_test.shape: ", X_test.shape)
    print("y_train.shape: ", y_train.shape)
    print("y_test.shape: ", y_test.shape)
    
    if not load_pretrained:
        print("Model Training has started")
        ## Training the autoencoder network
        autoencoder = autoencoder_model(size)
        autoencoder.fit(noisy_train_data, X_train_scaled,
                    epochs=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(noisy_test_data, X_test_scaled))
        autoencoder.save("saved_models/cdae_pretrained")
        
        ## 
        auto_output_train = autoencoder.predict(noisy_train_data)
        auto_output_test = autoencoder.predict(noisy_test_data)
        
        ## Training the cnn network
        cnn = cnn_model(size)
        cnn.fit(
            x=auto_output_train, 
            y=y_train, 
            batch_size=100, 
            epochs=128, 
            shuffle='true', 
            validation_data=(auto_output_test, y_test)
        )
        cnn.save("saved_models/cnn_pretrained")
        print("Training successfully completed")
        
        
    else:
        print("Loading pretrained models")
        autoencoder = load_model('saved_models/cdae_pretrained')
        cnn = load_model('saved_models/cnn_pretrained')
        print("Pretrained models successfully loaded")
    
    ## Testing a simple radiation scenario
     
    ## Creating an input to the framework with an AoI of 100mx100m and 200 sensors
    print()
    print("Initializing test aoi with 200 sensors")
    test_source_location = np.random.rand(1, 2) * (size**2)
    print("Source Location: ", test_source_location)
    test_sensor_locations = np.random.rand(200, 2) * (size**2)
    print(test_sensor_locations.shape)
    test_aoi = np.empty([size,size])
    for loc in test_sensor_locations:
        i = int(loc[0] / 100)
        j = int(loc[1] / 100)
        test_aoi[i][j] = (source_intensity * detector_area * efficiency) / (math.dist(loc, test_source_location.reshape(2))**2)
        
    ## Mean reconstruction
    window_size = 10
    while(np.isnan(test_aoi).any()):
        test_aoi = mean_reconstruction(test_aoi, window_size)
        window_size += 1
        
    ## Anomaly correction using pretrained autoencoder
    test_autoencoder_output = autoencoder.predict(test_aoi.reshape(1, size, size, 1))
    
    ## Threshold setting
    
    
    ## Localization using pretrained CNN
    test_cnn_output = cnn.predict(test_autoencoder_output)
    pred = test_cnn_output.reshape(size,size)
    print("Predicted source location: ", np.where(pred == np.amax(pred)))
    plot_grid(sensor_locations, test_cnn_output[0], size)
    
        
    

if __name__=='__main__':
    main()
