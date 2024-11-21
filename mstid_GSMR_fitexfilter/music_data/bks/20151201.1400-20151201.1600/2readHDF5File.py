# Name: Nicholas Gurra
# Advisors: Paul Jackowitz & Nathaniel Frissell
# Master of Software Engineering Thesis
# Fall 2024

# This file reads and loads in an encoded HDF5 file and prints its
# datasets, data, and groups. 

import h5py

hdf5_file_path = 'converted_data.h5'
output_file_path = 'hdf5_result.txt'

def print_hdf5_structure(name, obj, output_file):
    # Process the HDF5 structure and write to file.
    if isinstance(obj, h5py.Dataset):
        output_file.write(f"Dataset: {name}, Shape: {obj.shape}\n")
        output_file.write(f"Data: {obj[...]}\n\n")
    elif isinstance(obj, h5py.Group):
        output_file.write(f"Group: {name}\n")

def visit_and_write_to_file(name, obj):
    # Helper function for visiting items and writing them to a file.
    with open(output_file_path, 'a') as output_file:
        print_hdf5_structure(name, obj, output_file)

with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    with open(output_file_path, 'w') as output_file:
        output_file.write("HDF5 File Contents:\n")
    hdf5_file.visititems(visit_and_write_to_file)

print(f"The HDF5 structure and data have been saved to {output_file_path}")