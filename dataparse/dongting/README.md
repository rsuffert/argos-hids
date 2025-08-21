# DongTing Dataset Pre-processing for ARGOS-HIDS

This folder contains scripts and resources for pre-processing the DongTing dataset. DongTing dataset and all its auxiliary files can be downloaded from [here](https://zenodo.org/records/6627050).

## Usage

1. **Install dependencies**
     Ensure you have Python 3.x and required packages installed. You can install dependencies with:
    ```bash
    poetry install
    ```

2. **Set required environment variables**
    The following environment variables must be set so the script finds the required DongTing dataset files:
    ```bash
    export SYSCALL_TBL_PATH=<path>
    export NORMAL_DATA_FOLDER_PATH=<path>
    export ABNORMAL_DATA_FOLDER_PATH=<path>
    export BASELINE_XLSX_PATH=<path>
    ```

3. **Run the dataset loader/pre-processing script**
    After running the script with the below command, a `.h5` compressed file will be created with the syscall sequences for each label-split pair of the DongTing dataset. These can then be loaded to train the intrusion detection model.
    ```bash
    poetry run python3 loader.py
    ```