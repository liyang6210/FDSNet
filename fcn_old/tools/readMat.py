import scipy.io


def read_and_print_mat_file(file_path):
    """
    This function loads a .mat file and prints all the variables it contains,
    excluding standard MATLAB metadata like __header__, __version__, and __globals__.
    """
    try:
        # Load the .mat file
        mat_data = scipy.io.loadmat(file_path)

        # Iterate through all keys in the dictionary
        for key in mat_data:
            if key not in ['__header__', '__version__', '__globals__']:
                # Print each key and its corresponding data
                print(f"Variable name: {key}")
                print("Data:")
                print(mat_data[key])
                print("\n")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage example
# To use this script, simply replace 'your_mat_file.mat' with the path to your .mat file
read_and_print_mat_file('D:\\data_set\\UNIMIB2016原版\\annotations.mat')
