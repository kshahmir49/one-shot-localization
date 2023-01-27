import numpy as np

"""
This method is used to visualize the input or output in the form of grids
"""
def mean_reconstruction(image, window_size):
    for i in range(image.shape[0]-window_size+1):
        for j in range(image.shape[1]-window_size+1):
            window = image[i:i+window_size, j:j+window_size]
            if np.isnan(window).any():
                window = np.nan_to_num(window, nan=np.nanmean(window))
            image[i:i+window_size, j:j+window_size] = window
    return image