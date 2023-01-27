import numpy as np
import matplotlib.pyplot as plt


"""
This method is used to visualize the input or output in the form of grids
"""
def plot_grid(source_locations,array,size):
    arr = np.copy(array) 
    fig, ax = plt.subplots()
    ax.grid(which='major', color='black', linestyle='-', linewidth=1)
    # To display the values
    ax.set_xlim([0,size**2])
    ax.set_ylim([0,size**2])
    for [i,j], z in zip(source_locations,arr.ravel()):
        ax.text(i+0.5, j+0.5, '{:0.1f}'.format(z), ha='center', va='center')
    plt.xticks(np.arange(0, size**2, size))
    plt.yticks(np.arange(0, size**2, size))
    plt.show()
