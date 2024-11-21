import pickle
import numpy as np
from pyDARNmusic.music.music_array import musicArray

original_pickle_path = 'bks-20151201.1400-20151201.1600.p'
reconstructed_pickle_path = 'reconstructed_data.p'
comparison_file_path = 'comparison.txt'

# Load the original pickled musicArray object
with open(original_pickle_path, 'rb') as pickle_file:
    original_data_obj = pickle.load(pickle_file)

# Load the reconstructed pickled musicArray object
with open(reconstructed_pickle_path, 'rb') as pickle_file:
    reconstructed_data_obj = pickle.load(pickle_file)

# Check if two arrays are equal, handling NaNs as equal.
def arrays_are_equal_with_nan(arr1, arr2):
    if arr1.shape != arr2.shape:
        return False
    return np.all(np.isnan(arr1) == np.isnan(arr2)) and np.all(np.nan_to_num(arr1) == np.nan_to_num(arr2))

# Compare two musicArray objects and save the result to a file
def compare_music_array(original, reconstructed, output_file):
    with open(output_file, 'w') as f:
        # Compare PRM data
        if original.prm != reconstructed.prm:
            f.write("PRM data does not match.\n")
            f.write("Original PRM: {}\n".format(original.prm))
            f.write("Reconstructed PRM: {}\n".format(reconstructed.prm))
        else:
            f.write("PRM data matches.\n")

        # Compare messages
        if original.messages != reconstructed.messages:
            f.write("Messages do not match.\n")
            f.write("Original Messages: {}\n".format(original.messages))
            f.write("Reconstructed Messages: {}\n".format(reconstructed.messages))
        else:
            f.write("Messages match.\n")

        # Compare datasets
        original_datasets = original.get_data_sets()
        reconstructed_datasets = reconstructed.get_data_sets()
        
        if original_datasets != reconstructed_datasets:
            f.write("Data sets do not match.\n")
            f.write("Original Data Sets: {}\n".format(original_datasets))
            f.write("Reconstructed Data Sets: {}\n".format(reconstructed_datasets))
        else:
            f.write("Data sets match.\n")

        # Compare individual datasets
        for dataset_name in original_datasets:
            original_data = getattr(original, dataset_name)
            reconstructed_data = getattr(reconstructed, dataset_name, None)

            if reconstructed_data is None:
                f.write(f"Dataset '{dataset_name}' not found in reconstructed data.\n")
                continue

            # Compare time arrays
            if not np.array_equal(original_data.time, reconstructed_data.time):
                f.write(f"Time arrays in dataset '{dataset_name}' do not match.\n")
                f.write("Original Time: {}\n".format(original_data.time))
                f.write("Reconstructed Time: {}\n".format(reconstructed_data.time))
            else:
                f.write(f"Time arrays in dataset '{dataset_name}' match.\n")

            # Compare data arrays
            if not arrays_are_equal_with_nan(original_data.data, reconstructed_data.data):
                f.write(f"Data arrays in dataset '{dataset_name}' do not match.\n")
                f.write("Original Data: {}\n".format(original_data.data))
                f.write("Reconstructed Data: {}\n".format(reconstructed_data.data))
            else:
                f.write(f"Data arrays in dataset '{dataset_name}' match.\n")

            # Still working on comparing music data obj's here

        # This is included to see the attributes associated with both musicArrays
        f.write("\nOriginal Data Attributes:\n")
        for attr in dir(original):
            # Skip active and get_data_sets
            if not attr.startswith('__') and attr not in ['active', 'get_data_sets']:
                f.write(f"{attr}: {getattr(original, attr)}\n")

        f.write("\nReconstructed Data Attributes:\n")
        for attr in dir(reconstructed):
            if not attr.startswith('__') and attr not in ['active', 'get_data_sets']:
                f.write(f"{attr}: {getattr(reconstructed, attr)}\n")

compare_music_array(original_data_obj, reconstructed_data_obj, comparison_file_path)
print("Comparison completed and saved to", comparison_file_path)