import tempfile
import os
import numpy as np
import warnings

class TempStorage:
    """Manages temporary files for large datasets."""

    def __init__(self, prefix="Funcoin_"):
        # Create a temporary directory object
        self._temp_dir_obj = tempfile.TemporaryDirectory(prefix=prefix)
        self._temp_dir = self._temp_dir_obj.name  # Path to directory
        print(f'Created a temporary folder at {self._temp_dir}')
        self._files = []

    def save_FC(self, name, array):
        """Save a FC matrix to a temporary file."""
        file_path = os.path.join(self._temp_dir, f"{name}.npy")

        checkfile = os.path.isfile(file_path)

        if not checkfile:
            np.save(file_path, array)
            self._files.append(file_path)
        else:
            file_path = None    
            warn_str = f"The ID, \'{name}\', provided for temporarily saving data is already in use. To avoid overwriting, the data was not saved."      
            print(warn_str)  
            warnings.warn(warn_str, stacklevel=3)

        return file_path

    def load_FC(self, file_path):
        """Load a FC matrix from a temporary file."""
        return np.load(file_path)

    def list_files(self):
        """List all currently saved temporary files."""
        return list(self._files)

    def temp_dir_path(self):
        """Return the path to the temporary directory."""
        return self._temp_dir