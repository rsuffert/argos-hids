# GNN Argos HIDS Models

This folder contains the implementation and related files for the Graph Neural Network (GNN) models used in the Argos HIDS project. This includes both a supervised GNN model and an unsupervised GNN Autoencoder model.

## Pre-Requisites

- This assumes that the DongTing dataset has been loaded and pre-processed with the [`dataset/dongting/loader.py`](../../dataset/dongting/loader.py) script and the pre-processed `.h5` files for the DongTing dataset have been generated according to that script.

## Usage of [`supervised.py`](./supervised.py)

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

4. **Set required environment variables:**
    ```bash
    export NORMAL_TRAIN_H5=<path>.h5
    export NORMAL_VALID_H5=<path>.h5
    export NORMAL_TEST_H5=<path>.h5
    export ATTACK_TRAIN_H5=<path>.h5
    export ATTACK_VALID_H5=<path>.h5
    export ATTACK_TEST_H5=<path>.h5
    export SYSCALL_MAPPING_PATH=<path>.csv # the syscall names-to-IDs mapping outputted by the dataset pre-processing scripts
    ```

5. **Run the compatibility step for extracting and putting the H5 files in the correct structure:**
    ```bash
    poetry run python3 supervised.py -e
    ```

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
poetry run python3 supervised.py -e -p -i -t
```

For further information on the script parameters, run:

```bash
poetry run python3 supervised.py --help
```

## Usage of [`autoencoder.py`](./autoencoder.py) (unsupervised)

1. **Set up the environment (if not done already):**
    Follow steps 1-3 from the [`supervised.py` script usage](#usage-of-supervisedpy) section.
2. **Extract and pre-process the datasets (if not done already):** Follow steps 4-6 from the [`supervised.py` script usage](#usage-of-supervisedpy) section.
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