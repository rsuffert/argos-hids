# LID-DS Loader

A pipeline for converting LID-DS syscall datasets into ML-ready HDF5 format. This loader processes syscall traces from LID-DS scenarios and creates high-signal sequences optimized for intrusion detection training.


- **Attack data**: Extracts syscalls starting from the attack timestamp onward, capturing the malicious behavior pattern
- **Normal data**: Uses sliding window approach to process entire files into sequential 1024-syscall windows, ensuring comprehensive coverage of normal behavior

The processed data is automatically split into training/validation/test sets (60%/20%/20%) and saved in HDF5 format with gzip compression, maintaining compatibility with the DongTing dataset format for seamless integration across ARGOS components.

## Dataset

All scenarios from the LID dataset are available [here](https://drive.proton.me/urls/BWKRGQK994#fCK9JKL93Sjm).

## Usage

Considering your current working directory is the directory where this README file is located.

### 1. Install dependencies
Ensure you have Python 3.x and required packages installed. You can install dependencies with:
```bash
poetry install
```

### 2. Set required environment variables
The following environment variables must be set so the script finds the required dataset files:
```bash
# Directory where the LID decompressed scenarios directories are and where the preprocessed .h5 files will be stored
export LID_DATA_DIR=<path_to_lid_data>
```

### 3. Add the DongTing module to `PYTHONPATH`
This module imports functions from the DongTing module for consistent H5 formatting across datasets:
```bash
export PYTHONPATH=$(pwd)/../..
```

### 4. Run the dataset loader/preprocessing script
After running the script with the below command, `.h5` compressed files will be created with the syscall sequences for each label-split pair of the LID-DS dataset under the directory you set for `LID_DATA_DIR`. These files can then be loaded to train the intrusion detection model.
```bash
python loader.py
```

## Output Files

The loader generates the following files in your `LID_DATA_DIR`:

- **`syscall_dict.pkl`**: Cached mapping of syscall names to integer IDs
- **`0_training.h5`**: Normal sequences for training
- **`1_training.h5`**: Attack sequences for training  
- **`0_validation.h5`**: Normal sequences for validation
- **`1_validation.h5`**: Attack sequences for validation
- **`0_test.h5`**: Normal sequences for testing
- **`1_test.h5`**: Attack sequences for testing

Each H5 file contains variable-length sequences stored with gzip compression under the "sequences" dataset, optimized for machine learning workflows.

## Configuration

Key parameters can be adjusted in the `LIDDatasetLoader` class:

- **`WINDOW_SIZE`**: Maximum syscall sequence length (default: 1024)
- **`TRAIN_RATIO`**: Training data proportion (default: 0.6)
- **`VAL_RATIO`**: Validation data proportion (default: 0.2)
- **`MIN_CHUNK_SIZE`**: Minimum sequence chunk size (default: 50)
