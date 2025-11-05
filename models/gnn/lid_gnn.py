"""
LID Dataset GNN Training Pipeline.

This module provides an end-to-end pipeline for processing LID-DS dataset files
and training GNN models for syscall sequence classification.
"""
import os
import sys
import h5py
import logging
import argparse
import subprocess
import multiprocessing
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

from lib.preprocessing.graph_preprocess_dataset import preprocess_dataset
from lib.data.dataset import load_dataset

# Dataset paths
DATA_DIR = os.getenv("DATA_DIR", ".")
DATASET_NAME = os.getenv("DATASET_NAME", "lid-ds")

# H5 file paths from environment or defaults
NORMAL_TRAIN_H5 = os.getenv("NORMAL_TRAIN_H5", "0_training.h5")
NORMAL_VALID_H5 = os.getenv("NORMAL_VALID_H5", "0_validation.h5")
NORMAL_TEST_H5  = os.getenv("NORMAL_TEST_H5",  "0_test.h5")
ATTACK_TRAIN_H5 = os.getenv("ATTACK_TRAIN_H5", "1_training.h5")
ATTACK_VALID_H5 = os.getenv("ATTACK_VALID_H5", "1_validation.h5")
ATTACK_TEST_H5  = os.getenv("ATTACK_TEST_H5",  "1_test.h5")

# Output directories
TRAIN_TRACES_DIR    = "traces_train"
INFER_TRACES_DIR    = "traces_infer"
PKL_TRACES_FILENAME = "processed_graphs.pkl"

GNN_TRAINING_SCRIPT_PATH = os.path.join("lib", "train_graph.py")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the LID-GNN pipeline."""
    parser = argparse.ArgumentParser(description="Processing and training script for syscall graph classification")
    parser.add_argument("--extract", "-e", action="store_true", help="Run H5 to trace extraction")
    parser.add_argument("--preprocess_train", "-p", action="store_true", help="Trace to graph processing for training")
    parser.add_argument("--preprocess_infer", "-i", action="store_true", help="Trace to graph processing for inference")
    parser.add_argument("--train", "-t", action="store_true", help="Run graph model training")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Base directory containing H5 files")
    return parser.parse_args()


def parse_syscall_tbl(path: str) -> Dict[int, str]:
    """Parse syscall table file to create ID-to-name mapping."""
    syscalls_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    syscalls_map[int(parts[0])] = parts[2]
                except ValueError:
                    continue
    return syscalls_map


def _write_trace_file(syscall_names: list, trace_path: str) -> None:
    """Write syscall names to trace file."""
    with open(trace_path, "w", encoding="utf-8") as f:
        for name in syscall_names:
            f.write(f"{name}(\n")


def _process_sequence(
    seq: list, current_file_id: int, syscall_map: Dict[int, str], output_dir: str, files_per_subdir: int
) -> None:
    """Process a single sequence and write to trace file."""
    syscall_names = [syscall_map.get(int(sid), f"unknown_{sid}") for sid in seq]
    
    if len(syscall_names) < 2:
        return

    subdir_index = current_file_id // files_per_subdir
    subdir_path = os.path.join(output_dir, f"batch_{subdir_index}")
    os.makedirs(subdir_path, exist_ok=True)

    trace_path = os.path.join(subdir_path, f"trace_{current_file_id}.txt")
    _write_trace_file(syscall_names, trace_path)


def _process_h5_sequences(
    h5f: h5py.File, start_counter: int, syscall_map: Dict[int, str], output_dir: str, files_per_subdir: int
) -> int:
    """Process all sequences from H5 file."""
    if "sequences" not in h5f:
        logging.error("Dataset 'sequences' not found in H5 file")
        return start_counter

    sequences = h5f["sequences"]
    num_sequences = len(sequences)

    for i, seq in enumerate(sequences):
        _process_sequence(seq, start_counter + i, syscall_map, output_dir, files_per_subdir)

    return start_counter + num_sequences


def convert_h5_to_traces(h5_path: str,
                         output_dir: str,
                         syscall_tbl_path: str = "syscall_64.tbl",
                         files_per_subdir: int = 500,
                         start_counter: int = 0) -> int:
    """
    Convert sequences from H5 file to trace files organized in subdirectories.
    
    Args:
        h5_path (str): Path to input H5 file
        output_dir (str): Output directory for trace files
        syscall_tbl_path (str): Path to syscall table file
        files_per_subdir (int): Number of files per subdirectory
        start_counter (int): Starting counter for file naming
        
    Returns:
        int: Updated counter after processing all sequences
    """
    syscall_map = parse_syscall_tbl(syscall_tbl_path)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(h5_path):
        logging.error(f"H5 file not found: {h5_path}")
        return start_counter

    try:
        with h5py.File(h5_path, "r") as h5f:
            return _process_h5_sequences(h5f, start_counter, syscall_map, output_dir, files_per_subdir)
    except Exception as e:
        logging.error(f"Failed to process H5 file {h5_path}: {e}")
        return start_counter


def preprocess_traces_to_graphs_train() -> None:
    """Preprocess training traces to graph format."""
    output_filepath = preprocess_dataset(TRAIN_TRACES_DIR, False, False, PKL_TRACES_FILENAME)
    logging.info(f"Graphs saved to {output_filepath}")


def preprocess_traces_to_graphs_infer() -> None:
    """Preprocess inference traces to graph format using training vocabulary."""
    train_pkl_path = f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}"
    assert os.path.exists(train_pkl_path), \
        "Training dataset not found, please run preprocessing for training first."
    
    _, _, vocab = load_dataset(train_pkl_path)
    
    output_filepath = preprocess_dataset(
        INFER_TRACES_DIR, False, False, PKL_TRACES_FILENAME, vocab=vocab
    )
    logging.info(f"Graphs saved to {output_filepath}")


def train_gnn_model() -> None:
    """Train GNN model using preprocessed graphs."""
    assert os.path.exists(f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}"), \
        "Training dataset not found, please run preprocessing for training first."
    subprocess.run([
        "python", GNN_TRAINING_SCRIPT_PATH,
        "--dataset_path", f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}",
        "--model", "GIN",
        "--epochs", "250",
        "--batch_size", "256"
    ], check=True)


if __name__ == "__main__":
    args = parse_args()
    
    data_dir = args.data_dir

    if args.extract:
        logging.info("Converting H5 files to trace files...")
        
        def extract_normal() -> None:
            """Extract normal traces for training and inference."""
            counter = convert_h5_to_traces(os.path.join(data_dir, NORMAL_TRAIN_H5), f"{TRAIN_TRACES_DIR}/normal")
            counter = convert_h5_to_traces(
                os.path.join(data_dir, NORMAL_VALID_H5), 
                f"{TRAIN_TRACES_DIR}/normal", 
                start_counter=counter
            )
            convert_h5_to_traces(os.path.join(data_dir, NORMAL_TEST_H5), f"{INFER_TRACES_DIR}/normal")
        
        def extract_attack() -> None:
            """Extract attack traces for training and inference."""
            counter = convert_h5_to_traces(os.path.join(data_dir, ATTACK_TRAIN_H5), f"{TRAIN_TRACES_DIR}/attack")
            counter = convert_h5_to_traces(
                os.path.join(data_dir, ATTACK_VALID_H5), 
                f"{TRAIN_TRACES_DIR}/attack", 
                start_counter=counter
            )
            convert_h5_to_traces(os.path.join(data_dir, ATTACK_TEST_H5), f"{INFER_TRACES_DIR}/attack")
        
        procs = [multiprocessing.Process(target=f) for f in [extract_normal, extract_attack]]
        for p in procs: 
            p.start()
        for p in procs: 
            p.join()
        
        logging.info("H5 to trace extraction finished.")

    if args.preprocess_train:
        logging.info("Preprocessing traces to graphs for training...")
        preprocess_traces_to_graphs_train()

    if args.preprocess_infer:
        logging.info("Preprocessing traces to graphs for inference...")
        preprocess_traces_to_graphs_infer()

    if args.train:
        logging.info("Training GNN model...")
        train_gnn_model()