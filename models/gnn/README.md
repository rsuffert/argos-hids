# GNN Argos HIDS Models

This folder contains the implementation and related files for the Graph Neural Network (GNN) models used in the Argos HIDS project. This includes both a supervised GNN model and an unsupervised GNN Autoencoder model.

## Pre-Requisites

- This assumes that the dataset has been loaded and pre-processed with the appropriate loader script:
  - **DongTing dataset**: Use [`dataset/dongting/loader.py`](../../dataset/dongting/loader.py)
  - **LID-DS dataset**: Use [`dataparse/lid/loader.py`](../../dataparse/lid/loader.py)
- The pre-processed `.h5` files for your chosen dataset should be generated according to the respective loader script.

## Usage of [`supervised.py`](./supervised.py) (DongTing and LID-DS Datasets)

This script supports both the DongTing and LID-DS datasets by configuring the H5 file paths via environment variables or command-line arguments.

1. **Install dependencies:**
    ```bash
    poetry install
    ```

2. **Install PyTorch and PyTorch Geometric for your CUDA version:**
    PyTorch and PyTorch Geometric require special installation commands depending on your CUDA version (for GPU support).  
    **Follow these steps:**
    - Uninstall all PyTorch and related packages installed by Poetry:
        ```bash
        poetry run pip uninstall torch torchvision torchaudio torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -y
        ```

    - Go to the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) and select your OS, package manager (pip), language (Python), and CUDA version.  
      Copy the `pip install` command for PyTorch.

    - Then, go to the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), select your PyTorch and CUDA version, and copy the recommended `pip install` commands for PyTorch Geometric and its dependencies.

    - **Run all these commands inside your Poetry environment** (prepend `poetry run` to each command).  
      For example:
      ```bash
      poetry run pip install torch==<version> torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
      poetry run pip install torch_geometric
      poetry run pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-<version>+cuXXX.html
      ```
      Replace `<version>` and `cuXXX` with the versions matching your system.

    > **Note:** If you are using CPU only, select "CPU" on both websites.

3. **Add the GNN lib base directory to `PYTHONPATH`:**
    Needed for the training script to find the library.
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/lib
    ```

4. **Set required environment variables (optional, for customization):**
    You can customize the H5 file names and paths using environment variables. Defaults are set for DongTing; override for LID-DS as needed. The defaults are for DongTing dataset.
    ```bash
    export NORMAL_TRAIN_H5=<path>.h5  
    export NORMAL_VALID_H5=<path>.h5  
    export NORMAL_TEST_H5=<path>.h5   
    export ATTACK_TRAIN_H5=<path>.h5  
    export ATTACK_VALID_H5=<path>.h5  
    export ATTACK_TEST_H5=<path>.h5   
    export SYSCALL_MAPPING_PATH=<path>.csv  # Required for the syscall names-to-IDs mapping outputted by the dataset pre-processing scripts
    ```

5. **Run the compatibility step for extracting and putting the H5 files in the correct structure:**
    ```bash
    poetry run python3 supervised.py -e -d /path/to/dataset
    ```
    Replace `/path/to/dataset` with the directory containing your H5 files.

6. **Pre-process the extracted traces for training and then for inference:**
    Training pre-processing (`-p`) must run before inference pre-processing (`-i`) because the same vocabulary applied for the training data needs to be applied for the inference data.
    ```bash
    poetry run python3 supervised.py -p -i
    ```

7. **Train the model:**
    ```bash
    poetry run python3 supervised.py -t
    ```

**NOTE:** You may as well put it all in a single command. But keep in mind that you do not need to run with `-e`, `-p`, and `-i` every time you want to train the model.

```bash
poetry run python3 supervised.py -e -p -i -t -d /path/to/dataset
```

For further information on the script parameters, run:

```bash
poetry run python3 supervised.py --help
```

## Usage of [`lid_gnn.py`](./lid_gnn.py) (LID-DS Dataset)

1. **Install dependencies:**
    ```bash
    poetry install
    ```

2. **Install PyTorch and PyTorch Geometric:**
    Follow step 2 from the [`supervised.py` script usage](#usage-of-supervisedpy-dongting-and-lid-ds-datasets) section above.

3. **Add the GNN lib base directory to `PYTHONPATH`:**
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/lib
    ```

4. **Extract H5 files to trace format:**
    The LID-DS loader creates H5 files and a `syscall_dict.pkl` mapping file. Point to the directory containing these files:
    ```bash
    poetry run python3 lid_gnn.py --extract -d /path/to/lid-data
    ```
    This will:
    - Read the H5 files (`0_training.h5`, `0_validation.h5`, `0_test.h5`, `1_training.h5`, `1_validation.h5`, `1_test.h5`)
    - Use `syscall_dict.pkl` to convert syscall IDs to names
    - Create trace files in `traces_train/` and `traces_infer/` directories

5. **Pre-process the extracted traces:**
    Training pre-processing must run before inference pre-processing:
    ```bash
    poetry run python3 lid_gnn.py --preprocess_train
    poetry run python3 lid_gnn.py --preprocess_infer
    ```

6. **Train the model:**
    ```bash
    poetry run python3 lid_gnn.py --train
    ```

**NOTE:** You can run all steps in a single command:
```bash
poetry run python3 lid_gnn.py --extract --preprocess_train --preprocess_infer --train -d /path/to/lid-data
```

For further information on the script parameters, run:
```bash
poetry run python3 lid_gnn.py --help
```

**Environment Variables (Optional):**
You can customize the H5 file names and paths using environment variables:
```bash
export DATA_DIR=/path/to/lid-data
export NORMAL_TRAIN_H5=0_training.h5
export NORMAL_VALID_H5=0_validation.h5
export NORMAL_TEST_H5=0_test.h5
export ATTACK_TRAIN_H5=1_training.h5
export ATTACK_VALID_H5=1_validation.h5
export ATTACK_TEST_H5=1_test.h5
```

## Usage of [`autoencoder.py`](./autoencoder.py) (Unsupervised, DongTing and LID-DS Datasets)

This script supports both the DongTing and LID-DS datasets by using the processed graph files from either `supervised.py` or `lid_gnn.py`.

1. **Set up the environment (if not done already):**
    Follow steps 1-3 from the [`supervised.py` script usage](#usage-of-supervisedpy-dongting-and-lid-ds-datasets) section.
2. **Extract and pre-process the datasets (if not done already):** 
    Follow the extraction and preprocessing steps from either the [`supervised.py`](#usage-of-supervisedpy-dongting-and-lid-ds-datasets) or [`lid_gnn.py`](#usage-of-lid_gnnpy-lid-ds-dataset) sections depending on your dataset.
3. **Run the training script:**
    ```bash
    poetry run python3 autoencoder.py \
        --train_dataset traces_train/processed_graphs.pkl \
        --test_dataset traces_infer/processed_graphs.pkl \
        --epochs 50 \
        --batch_size 64 \
        --learning_rate 0.001 \
        --target_fpr 0.05 \
        --num_layers 4
    ```
    Note that the parameters are a suggestion and can be customized. For information on what parameters can be supplied, run:
    ```bash
    poetry run python3 autoencoder.py --help
    ```