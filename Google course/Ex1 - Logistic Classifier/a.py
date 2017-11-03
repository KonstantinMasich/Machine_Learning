import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

IMAGE_SIZE  = 28     # Pixel width and height.
PIXEL_DEPTH = 255.0  # Number of levels per pixel.

def load_letter(folder, images_required=-1):
    """Load the data for a single letter label. Loads images_required images!!"""
    image_files = os.listdir(folder)
    # If num of required images is not specified - load them all:
    if images_required <= 0:
        images_required = len(image_files)
    dataset = np.ndarray(shape=(images_required, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    print("===============\nNow working in folder:", folder,"\n===============")
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            # 1. Load image data as 2D array (shaped 28x28) of pixel values
            image_data = (ndimage.imread(image_file).astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            # 2. Load data into dataset and move dataset "iterator" a step ahead
            dataset[num_images, :, :] = image_data
            num_images += 1
            # --- To show images:
            #imgplot = plt.imshow(image_data)
            #plt.show()
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        if num_images >= images_required:
                break
    dataset = dataset[0:num_images, :, :]
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    print(images_required)
    return dataset

"""   
def maybe_pickle(data_folders, required_num_images_per_class, force=False):
    # To see contents of pickle file, type in terminal:
    # python3 -mpickle name_of_pickle_file.pickle
    dataset_names = []
    for folder in os.listdir(data_folders):
        full_folder_name = data_folders + "/" + folder
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, required_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
        return dataset_names
"""

def maybe_pickle2(samples_folder, pickle_folder, required_num_images_per_class, force=False):
    # To see contents of pickle file, type in terminal:
    # python3 -mpickle name_of_pickle_file.pickle
    dataset_names = []
    for folder in os.listdir(samples_folder):
        set_filename = pickle_folder + folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s' % set_filename)
            samples = samples_folder + folder + "/"
            dataset = load_letter(samples, required_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names
    
def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels  = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


# 1. Pickling parts
samples_folder = "mnist/testing/"
pickles_folder = "mnist/pickles/testing/"
maybe_pickle2(samples_folder, pickles_folder, -1)
samples_folder = "mnist/training/"
pickles_folder = "mnist/pickles/training/"
maybe_pickle2(samples_folder, pickles_folder, -1)


