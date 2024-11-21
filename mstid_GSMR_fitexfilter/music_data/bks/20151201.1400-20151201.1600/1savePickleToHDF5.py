# Name: Nicholas Gurra
# Advisors: Paul Jackowitz & Nathaniel Frissell
# Master of Software Engineering Thesis
# Fall 2024

# This file takes in a musicArray pickle file, processes all of its attributes and data,
# assigns them into groups, datasets and data, and converts it to HDF5.

import pickle
import h5py
import numpy as np
from datetime import datetime, timedelta
import json

pickle_file_path = 'bks-20151201.1400-20151201.1600.p'
hdf5_file_path = 'converted_data.h5'

def serializeToString(obj):
    """Serialize inputted objects to strings for HDF5 storage."""
    if isinstance(obj, (int, float, str)):
        # Handle scalar values
        return obj
    elif isinstance(obj, datetime):
        # Convert datetime to ISO string
        return obj.isoformat()
    elif isinstance(obj, list):
        # Recursively serialize list elements
        return [serializeToString(item) for item in obj]
    elif isinstance(obj, dict):
        # Recursively serialize dictionary
        return {serializeToString(key): serializeToString(value) for key, value in obj.items()}
    else:
        # Default to string representation
        return str(obj)

def saveDictToHDF5(hdf5_group, data_dict):
    """Save a dictionary's data to an HDF5 group."""
    for key, values in data_dict.items():
        try:
            # Intentionally ignore parent attribute
            if key == "parent":
                continue

            if key == "fov":
                if key not in hdf5_group:
                    # Handle datetimes
                    if isinstance(values, datetime):
                        hdf5_group.create_dataset(key, data=np.string_(values.isoformat()))
                        continue

                    # Handle lists and convert to arrays
                    if isinstance(values, list):
                        array_data = np.array(values)
                        hdf5_group.create_dataset(key, data=array_data)

                    # Handle NumPy arrays
                    elif isinstance(values, np.ndarray):
                        hdf5_group.create_dataset(key, data=values)

                    # Handle dictionaries
                    elif isinstance(values, dict):
                        sub_group = hdf5_group.create_group(key)
                        saveDictToHDF5(sub_group, values)

                    # Handle scalar values (int, float, str)
                    elif isinstance(values, (int, float, str)):
                        hdf5_group.create_dataset(key, data=np.string_(values) if isinstance(values, str) else values)
                    # Kept for testing
                    else:
                        print(f"Error: Type not handled for key '{key}'. Type: {type(values)}")

            # Handle datetimes
            elif isinstance(values, datetime):
                values = serializeToString(values)
                hdf5_group.create_dataset(key, data=np.string_(values))
                continue

            # Handle list of datetimes
            elif isinstance(values, list) and all(isinstance(v, datetime) for v in values):
                values = [serializeToString(v) for v in values]
                hdf5_group.create_dataset(key, data=np.array(values, dtype='S'))
                continue

            # Handle dictionaries
            elif isinstance(values, dict):
                sub_group = hdf5_group.create_group(key)
                serialized_values = serializeToString(values)
                saveDictToHDF5(sub_group, serialized_values)

            # Handle lists
            elif isinstance(values, list):
                serialized_values = serializeToString(values)
                hdf5_group.create_dataset(key, data=np.array(serialized_values, dtype='S'))

            # Handle numpy ndarrays
            elif isinstance(values, np.ndarray):
                if values.dtype == 'O':
                    serialized_values = np.array([serializeToString(item) for item in values], dtype='S')
                    hdf5_group.create_dataset(key, data=serialized_values)
                else:
                    hdf5_group.create_dataset(key, data=values)

            elif isinstance(values, bool):
                hdf5_group.create_dataset(key, data=values)

            elif isinstance(values, int):
                hdf5_group.create_dataset(key, data=values)

            else:
                # Default to serializing the object as a string
                serialized_value = serializeToString(values)
                hdf5_group.create_dataset(key, data=np.string_(serialized_value))

        except Exception as e:
            print(f"Could not save {key} in {hdf5_group.name}: {e}")



def saveMusicArrayToHDF5(hdf5_file, music_array_obj):
    """Save the contents of the musicArray object's __dict__ and DS objects to HDF5."""

    for attr_name in dir(music_array_obj):
        # Skip the 'active' attribute and any special methods
        if attr_name == 'active' or attr_name.startswith('__'):
            continue  

        data = getattr(music_array_obj, attr_name)
        # Check for dictionary type and create group (prm dictionary)
        if isinstance(data, dict):
            group = hdf5_file.create_group(attr_name)
            saveDictToHDF5(group, serializeToString(data))
        # Check for list type and create dataset (messages list)
        elif isinstance(data, list):
            hdf5_file.create_dataset(attr_name, data=np.array(data, dtype='S'))
        # Check for "DSXXX" object type and create group
        elif attr_name.startswith('DS'):
            ds_group = hdf5_file.create_group(f"{attr_name}")
            saveDictToHDF5(ds_group, data.__dict__)

with open(pickle_file_path, 'rb') as pickle_file:
    data = pickle.load(pickle_file)
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        saveMusicArrayToHDF5(hdf5_file, data)
    print(f"{pickle_file_path} has been converted to HDF5 and saved as {hdf5_file_path}")