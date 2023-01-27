import numpy as np
import math
import random


"""
This method is used to generate CPM values for each grid for a specified
source location
Variables:
coordinates - a list of various source location
size - size of AoI
grid_coordinates - the cartesian coordinates of each grid
"""
def generate_grid_values_from_sourceloc(coordinates, size, grid_coordinates, source_intensity, detector_area, efficiency):
    x = np.empty([coordinates.shape[0], size,size])
    grid_values = np.empty([size,size])
    for k in range(coordinates.shape[0]):
        for i in range(size):
            for j in range(size):
                # detector_area = random.uniform(0.0, 0.02)
                # efficiency = random.uniform(0.0, 1)
                x[k][i][j] = (source_intensity * detector_area * efficiency) / (math.dist(grid_coordinates[i][j], coordinates[k])**2)
    return x

"""
This method is used to generate the cartesian coordinates
of each grid in the AoI
"""
def generate_grid_coordinates(size):
    grid_coordinates = np.zeros((size,size,2))
    for i in range(size):
        for j in range(size):
            grid_coordinates[i][j][0] = ((i+1) * size) - size/2
            grid_coordinates[i][j][1] = ((j+1) * size) - size/2
    return grid_coordinates


"""
This method is used to assign a grid for a source location (which is originally in coordinates).
This method is used to generate training variable y which has 0s and 1s
1 denoting the source location.
"""
def generate_sourceloc_from_coordinates(coordinates, size):
    y = np.empty([coordinates.shape[0], size,size])
    for k in range(coordinates.shape[0]):
        i = int(coordinates[k][0] / size)
        j = int(coordinates[k][1] / size)
        y[k][i][j] = 1
    return y

"""
This method is used to add anomalies to grids
"""
def add_anomaly(array, no_of_anomalies, size):

    noise_array = np.copy(array)

    for i in range(array.shape[0]):
        for j in range(no_of_anomalies):
            noise_value = random.randint(0,10000)
            random_row_index = random.randint(0,size-1)
            random_column_index = random.randint(0,size-1)
            noise_array[i][random_row_index][random_column_index] = noise_value

    return noise_array
