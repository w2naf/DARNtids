import pickle
import h5py
import numpy as np
import datetime
from pyDARNmusic import load_fitacf
from pyDARNmusic.music.music_array import musicArray

pickle_file_path = 'bks-20151201.1400-20151201.1600.p'
hdf5_file_path = 'converted_data.h5'
reconstructed_file_path = 'reconstructed_data.p'

# # Load the original pickled musicArray object
# with open(pickle_file_path, 'rb') as pickle_file:
#     original_data_obj = pickle.load(pickle_file)

# Extract parameters from the HDF5 file
with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    radar = hdf5_file['DS000_originalFit/metadata/code'][()].decode('utf-8').strip()
    sTime = datetime.datetime.strptime(hdf5_file['DS000_originalFit/metadata/sTime'][()].decode('utf-8'), '%Y-%m-%dT%H:%M:%S.%f')
    eTime = datetime.datetime.strptime(hdf5_file['DS000_originalFit/metadata/eTime'][()].decode('utf-8'), '%Y-%m-%dT%H:%M:%S.%f')
    data_dir = '/sd-data'
    fovModel = hdf5_file['DS000_originalFit/metadata/model'][()]
    gscat = hdf5_file['DS000_originalFit/metadata/gscat'][()]

# Load fitacf data
fitacf = load_fitacf(radar, sTime, eTime, data_dir=data_dir)

# Create a reconstructed musicArray object with the extracted parameters
reconstructed_data_obj = musicArray(fitacf, sTime=sTime, eTime=eTime, fovModel=fovModel, gscat=gscat)

# Save the reconstructed object to a new pickle file
with open(reconstructed_file_path, 'wb') as pickle_file:
    pickle.dump(reconstructed_data_obj, pickle_file)

print(f"Reconstructed data object saved as {reconstructed_file_path}.")
