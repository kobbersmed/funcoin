import tempfile
import os
import numpy as np
import warnings

class TempStorage:
    """Manages temporary files for large datasets."""

    def __init__(self, dir = None, prefix="Funcoin_", tempdata = True):
        # Create a temporary directory object
        if tempdata:
            self._temp_dir_obj = tempfile.TemporaryDirectory(prefix=prefix, dir=dir)
            self._temp_dir = self._temp_dir_obj.name  # Path to directory
            print(f'Created a temporary folder at {self._temp_dir}')
        
        self._tempdata = tempdata
        self._files = []
        self._datatype = None

    def save_FC(self, ID, FC):
        """Save a FC matrix to a temporary file."""
        file_path = os.path.join(self._temp_dir, f"{ID}.npy")

        checkfile = os.path.isfile(file_path)

        if not checkfile:
            np.save(file_path, FC)
            self._files.append(file_path)
        else:
            file_path = None    
            warn_str = f"The ID, \'{ID}\', provided for temporarily saving data is already in use. To avoid overwriting, the data was not saved."      
            warnings.warn(warn_str, stacklevel=3)

        return file_path

    def load_FC(self, file_path):
        """Load a FC matrix from a temporary file."""
        FC_here = np.load(file_path)
        if self._datatype == 'FC':
            FC = FC_here
        elif self._datatype == 'FC_eigen':
            FC = FC_here@FC_here.T
        return FC
    
    def list_files(self):
        """List all currently saved temporary files."""
        return list(self._files)

    def temp_dir_path(self):
        """Return the path to the temporary directory."""
        return self._temp_dir
    
    def cleanup(self):
        print(f"Cleaning up {self._temp_dir_obj.name}")
        self._temp_dir_obj.cleanup()