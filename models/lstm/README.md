# LSTM ARGOS-HIDS Model

This folder contains the implementation and related files for the LSTM (Long Short-Term Memory) model used in the Argos-HIDS project. The `supervised.py` script contains the implementation for the training of a regular LSTM model, whereas the `autoencoder.py` script is the training for a unsupervised LSTM autoencoder.

## Pre-Requisites

- This assumes that the DongTing dataset has been loaded and pre-processed with the [`dataset/dongting/loader.py`](../../dataset/dongting/loader.py) script and the pre-processed `.h5` files for the DongTing dataset have been generated according to that script.

## Usage

1. **Install dependencies:**
    ```bash
    poetry install
    ```

2. **Set required environment variables**
    ```bash
    # These should point to the DongTing normal/attack train/validation splits
    # They are generated after running the previously mentioned script
    export NORMAL_TRAIN_DT_PATH=<path>
    export ATTACK_TRAIN_DT_PATH=<path>
    export NORMAL_VALID_DT_PATH=<path>
    export ATTACK_VALID_DT_PATH=<path>
    ```

3. **Train the model:**
    ```bash
    # to train the supervised LSTM
    poetry run python3 supervised.py

    # to train the LSTM autoencoder
    poetry run python3 autoencoder.py
    ```

The trainer scripts will take care of automatically saving the trained model to the `./lstm.pt` and `./lstm-autoencoder.pt` files.