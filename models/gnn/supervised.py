"""Training script for a supervised GNN model to be used for the Argos HIDS project."""

import os
import h5py
import sys
import logging
import argparse
import subprocess
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

from lib.data.dataset import load_dataset
from lib.preprocessing.graph_preprocess_dataset import preprocess_dataset

NORMAL_TRAIN_H5 = os.getenv("NORMAL_TRAIN_H5", "0_training.h5")
NORMAL_VALID_H5 = os.getenv("NORMAL_VALID_H5", "0_validation.h5")
NORMAL_TEST_H5  = os.getenv("NORMAL_TEST_H5",  "0_test.h5")
ATTACK_TRAIN_H5 = os.getenv("ATTACK_TRAIN_H5", "1_training.h5")
ATTACK_VALID_H5 = os.getenv("ATTACK_VALID_H5", "1_validation.h5")
ATTACK_TEST_H5  = os.getenv("ATTACK_TEST_H5",  "1_test.h5")

TRAIN_TRACES_DIR    = "traces_train"
INFER_TRACES_DIR    = "traces_infer"
PKL_TRACES_FILENAME = "processed_graphs.pkl"

GNN_TRAINING_SCRIPT_PATH = os.path.join("lib", "train_graph.py")

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the end-to-end processing and training script.

    Returns: argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="End-to-end processing and training script for syscall graph classification")
    parser.add_argument("--extract", "-e", action="store_true", help="Run H5 to trace extraction")
    parser.add_argument("--preprocess_train", "-p", action="store_true",
                        help="Run trace to graph preprocessing for training the model")
    parser.add_argument("--preprocess_infer", "-i", action="store_true",
                        help="Run trace to graph preprocessing for inference with the trained model")
    parser.add_argument("--train", "-t", action="store_true", help="Run graph model training")
    parser.add_argument("--data_dir", "-d", type=str, default=".", help="Base directory containing H5 files")  # Added
    return parser.parse_args()

def convert_h5_to_traces(
    h5_path: str,
    output_dir: str
) -> None:
    """
    Converts sequences of syscall IDs from an H5 file into trace files with numeric IDs, space-separated on one line.

    Args:
        h5_path (str): Path to the input H5 file containing syscall sequences.
        output_dir (str): Directory where the generated trace files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    counter = 0  # Reset counter for batching
    with h5py.File(h5_path, "r") as h5f:
        sequences = h5f["sequences"]
        for seq in sequences:
            if len(seq) < 2:
                # skip sequences with one or zero syscalls as they cannot be
                # used to build a graph
                continue
            # Add batching: create subdirectories with 500 files each
            subdir_index = counter // 500
            subdir_path = os.path.join(output_dir, f"batch_{subdir_index}")
            os.makedirs(subdir_path, exist_ok=True)
            trace_path = os.path.join(subdir_path, f"trace_{counter}.txt")
            logging.debug(f"Writing {trace_path} with {len(seq)} syscalls")
            with open(trace_path, "w") as f:
                f.write(" ".join(str(id) for id in seq))
            counter += 1

def preprocess_traces_to_graphs_train() -> None:
    """Pre-processes the training data directory into a PKL graph file."""
    output_filepath = preprocess_dataset(TRAIN_TRACES_DIR, False, False, PKL_TRACES_FILENAME)
    print(f"Graphs saved to {output_filepath}")

def preprocess_traces_to_graphs_infer() -> None:
    """
    Pre-processes the inference data directory into a PKL graph file using the
    training data's vocabulary.
    """
    assert os.path.exists(f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}"), \
        "Training dataset not found, please run preprocessing for training first."
    _, _, vocab = load_dataset(f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}")
    output_filepath = preprocess_dataset(
        INFER_TRACES_DIR, False, False, PKL_TRACES_FILENAME, vocab=vocab
    )
    print(f"Graphs saved to {output_filepath}")

def train_gnn_model() -> None:
    """Trains a GNN model with the pre-processed PKL graph files."""
    assert os.path.exists(f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}"), \
        "Training dataset not found, please run preprocessing for training first."
    subprocess.run([
        "python", GNN_TRAINING_SCRIPT_PATH,
        "--dataset_path", f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}",
        "--model", "GIN",
        "--epochs", "25",
        "--batch_size", "256",
        "--save_model_path", "gnn.pt"
    ], check=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    args = parse_args()
    data_dir = args.data_dir  # for H5 files from both datasets
    if args.extract:
        logging.info("Converting H5 files to trace files...")
        def extract_normal() -> None:
            """Wrapper for extracting normal traces from H5 files."""
            convert_h5_to_traces(os.path.join(data_dir, NORMAL_TRAIN_H5), f"{TRAIN_TRACES_DIR}/normal")  
            convert_h5_to_traces(os.path.join(data_dir, NORMAL_VALID_H5), f"{TRAIN_TRACES_DIR}/normal")  
            convert_h5_to_traces(os.path.join(data_dir, NORMAL_TEST_H5),  f"{INFER_TRACES_DIR}/normal")  
        def extract_attack() -> None:
            """Wrapper for extracting attack traces from H5 files."""
            convert_h5_to_traces(os.path.join(data_dir, ATTACK_TRAIN_H5), f"{TRAIN_TRACES_DIR}/attack")  
            convert_h5_to_traces(os.path.join(data_dir, ATTACK_VALID_H5), f"{TRAIN_TRACES_DIR}/attack")  
            convert_h5_to_traces(os.path.join(data_dir, ATTACK_TEST_H5),  f"{INFER_TRACES_DIR}/attack")  
        normal_proc = multiprocessing.Process(target=extract_normal)
        attack_proc = multiprocessing.Process(target=extract_attack)
        normal_proc.start()
        attack_proc.start()
        normal_proc.join()
        attack_proc.join()
    if args.preprocess_train:
        logging.info("Preprocessing traces to graphs to train the model...")
        preprocess_traces_to_graphs_train()
    if args.preprocess_infer:
        logging.info("Preprocessing traces to graphs for inference with the trained model...")
        preprocess_traces_to_graphs_infer()
    if args.train:
        logging.info("Training GNN model...")
        train_gnn_model()