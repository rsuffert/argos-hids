# LID-DS Loader

A pipeline for converting LID-DS syscall datasets into ML-ready HDF5 format. This loader processes syscall traces from LID-DS scenarios. All scenarios from the LID dataset are available [here](https://drive.proton.me/urls/BWKRGQK994#fCK9JKL93Sjm).

## Usage

Considering your current working directory is the directory where this README file is located.

1. **Install dependencies**
     Ensure you have Python 3.x and required packages installed. You can install dependencies with:
    ```bash
    pip install -r requirements.txt
    ```

2. **Set required environment variables**
    The following environment variables must be set so the script finds the required DongTing dataset files:
    ```bash
    # Directory where the LID discompressed scenarios directories are and where the preprocessed .h5 files will be stored
    export LID_DATA_DIR=<path>

    # Path to the syscall tables that contains the mappings from syscall name to ID
    export SYSCALL_TBL_PATH=<path>
    ```

3. **Add the DongTing module to `PYTHONPATH`**
    This module imports functions from the DongTing module, so we need to tell Python where to find it.
    ```bash
    export PYTHONPATH=$(pwd)/../..
    ```

4. **Run the dataset loader/pre-processing script**
    After running the script with the below command, a `.h5` compressed file will be created with the syscall sequences for each label-split pair of the LID-DS dataset under the directory you set for `LID_DATA_DIR`. These can then be loaded to train the intrusion detection model.
    ```bash
    python loader.py
    ```