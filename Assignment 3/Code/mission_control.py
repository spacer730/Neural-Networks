"""
Contains all the variables necessary to run main.py file.
"""

# Set LOAD to True to load a trained model or set it False to train a new one.
LOAD = False

# Dataset directories. Use the second line for Hubble training data.
DATASET_PATH = './Dataset/SDSS/'
#DATASET_PATH = './Dataset/Hubble/'
DATASET_CHOSEN = 'galaxies'  # required by utils.py


# Model hyperparameters
Z_DIM = 100  # The input noise vector dimension
BATCH_SIZE = 12
N_ITERATIONS = 30000
LEARNING_RATE = 0.0002
BETA_1 = 0.5
IMAGE_SIZE = 64 # Change the Generator model if the IMAGE_SIZE needs to be changed to a different value
