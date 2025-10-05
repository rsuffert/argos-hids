"""
LID Dataset Loader for ARGOS-HIDS - Simplified for maintainability.

This module provides functionality to load and process syscall datasets in two formats:
1. LID-DS format: ZIP files containing .sc (syscall) and .json (metadata) files
2. H5 format: HDF5 files with preprocessed syscall sequences

The loader extracts syscall traces, splits them into training/inference sets,
and saves them as text files for further processing by GNN models.
"""

import os
import json
import logging
import argparse
import zipfile
import tempfile
import h5py
from pathlib import Path
from typing import List, Tuple

try:
    from preprocessing.graph_preprocess_dataset import preprocess_dataset # type: ignore
except ImportError:
    preprocess_dataset = None


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the syscall classification pipeline.
    
    Returns:
        argparse.Namespace: Parsed arguments containing dataset paths and options.
            - lidds: Path to LID-DS dataset directory
            - normal: Path to normal traces H5 file
            - attack: Path to attack traces H5 file  
            - preprocess: Flag to enable graph preprocessing
    """
    parser = argparse.ArgumentParser(description="Syscall Classification Pipeline")
    parser.add_argument("-l", "--lidds", type=str, help="Path to LID-DS dataset directory")
    parser.add_argument("-n", "--normal", type=str, help="Path to normal traces H5 file")
    parser.add_argument("-a", "--attack", type=str, help="Path to attack traces H5 file")
    parser.add_argument("-p", "--preprocess", action="store_true", help="Preprocess training data to graphs")
    return parser.parse_args()


def save_trace_file(syscalls: List[str], file_path: str) -> None:
    """
    Save a single syscall trace to a text file.
    
    Each syscall is written on a separate line with parentheses appended
    to match the expected format for downstream processing.
    
    Args:
        syscalls: List of syscall names to save
        file_path: Output file path where trace will be written
    """
    with open(file_path, "w") as f:
        f.write("\n".join(f"{sc}(" for sc in syscalls))


def process_traces(traces: List[List[str]], output_dir: str) -> int:
    """
    Process and save multiple syscall traces to a directory.
    
    Filters traces to only include those with at least 20 syscalls,
    then saves each valid trace as a separate numbered file.
    
    Args:
        traces: List of syscall trace lists to process
        output_dir: Directory where trace files will be saved
        
    Returns:
        int: Number of trace files successfully created
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_count = 0
    for trace in traces:
        if len(trace) >= 20:
            save_trace_file(trace, f"{output_dir}/trace_{file_count}.txt")
            file_count += 1
    
    return file_count


def load_h5_file(h5_path: str) -> List[List[str]]:
    """
    Load syscall traces from an HDF5 file.
    
    Expects the H5 file to contain a "sequences" dataset with numeric
    syscall IDs. Converts IDs to strings and filters out zero padding.
    
    Args:
        h5_path: Path to the HDF5 file to load
        
    Returns:
        List[List[str]]: List of syscall traces as string lists.
                        Returns empty list if file doesn't exist or on error.
    """
    if not h5_path or not os.path.exists(h5_path):
        return []
    
    try:
        with h5py.File(h5_path, "r") as f:
            data = f["sequences"]
            return [[str(int(s)) for s in data[i] if s != 0] for i in range(data.shape[0])]
    except Exception as e:
        logging.error(f"Error loading {h5_path}: {e}")
        return []


def process_h5_datasets(normal_path: str, attack_path: str) -> None:
    """
    Process normal and attack syscall traces from H5 format files.
    
    Loads traces from both files, splits them 70/30 for training/testing,
    and saves the processed traces to appropriate directories.
    
    Args:
        normal_path: Path to H5 file containing normal (benign) syscall traces
        attack_path: Path to H5 file containing attack (malicious) syscall traces
    """
    normal_traces = load_h5_file(normal_path)
    attack_traces = load_h5_file(attack_path)
    
    # Split 70/30
    normal_split = int(0.7 * len(normal_traces))
    attack_split = int(0.7 * len(attack_traces))
    
    # Save training data
    normal_count = process_traces(normal_traces[:normal_split], "traces_train/normal")
    attack_count = process_traces(attack_traces[:attack_split], "traces_train/attack")
    
    # Save test data
    test_normal_count = process_traces(normal_traces[normal_split:], "traces_infer/normal")
    test_attack_count = process_traces(attack_traces[attack_split:], "traces_infer/attack")
    
    logging.info(f"Train: {normal_count} normal, {attack_count} attack")
    logging.info(f"Test: {test_normal_count} normal, {test_attack_count} attack")


def extract_from_zip(zip_path: Path, temp_dir: str) -> Tuple[List[Path], dict]:
    """
    Extract syscall and metadata files from a ZIP archive.
    
    Extracts .sc (syscall) and .json (metadata) files while filtering out
    macOS system files (__MACOSX, .DS_Store). Loads JSON metadata into memory.
    
    Args:
        zip_path: Path to ZIP file to extract from
        temp_dir: Temporary directory for extracted files
        
    Returns:
        Tuple containing:
            - List of paths to extracted .sc files
            - Dictionary mapping file stems to loaded JSON metadata
    """
    syscall_files: List[Path] = []
    metadata: dict[str, dict] = {}
    
    if "__MACOSX" in str(zip_path) or ".DS_Store" in str(zip_path):
        return syscall_files, metadata
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.filelist:
                if "__MACOSX" in info.filename or ".DS_Store" in info.filename:
                    continue
                
                if info.filename.endswith(".sc"):
                    name = Path(info.filename).stem
                    out_path = Path(temp_dir) / f"{name}.sc"
                    out_path.write_bytes(zf.read(info))
                    syscall_files.append(out_path)
                    
                elif info.filename.endswith(".json"):
                    name = Path(info.filename).stem
                    out_path = Path(temp_dir) / f"{name}.json"
                    out_path.write_bytes(zf.read(info))
                    try:
                        with open(out_path) as f:
                            metadata[name] = json.load(f)
                    except Exception:
                        pass
    except Exception as e:
        logging.warning(f"Error with {zip_path}: {e}")
    
    return syscall_files, metadata


def parse_syscall_file(file_path: Path) -> List[str]:
    """
    Parse a syscall trace file and extract syscall names.
    
    Expects each line to contain space-separated values where the syscall
    name is either the first field (if only one field) or the second field
    (if multiple fields, assuming first is timestamp).
    
    Args:
        file_path: Path to the .sc file to parse
        
    Returns:
        List[str]: List of syscall names extracted from the file.
                  Returns empty list on parsing errors.
    """
    try:
        with open(file_path) as f:
            syscalls = []
            for line in f:
                parts = line.strip().split()
                if parts:
                    syscall = parts[1] if len(parts) >= 2 else parts[0]
                    syscalls.append(syscall)
            return syscalls
    except Exception as e:
        logging.warning(f"Error parsing {file_path}: {e}")
        return []


def process_lidds_dataset(dataset_path: str) -> None:
    """
    Process a complete LID-DS format dataset.
    
    Recursively finds all ZIP files in the dataset directory, extracts
    syscall traces and metadata, classifies traces as normal/attack based
    on metadata, and saves processed traces for training and inference.
    
    Args:
        dataset_path: Root directory containing LID-DS ZIP files
    """
    logging.info(f"Processing LID-DS: {dataset_path}")
    
    normal_traces = []
    attack_traces = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract all ZIP files
        for zip_path in Path(dataset_path).rglob("*.zip"):
            syscall_files, metadata = extract_from_zip(zip_path, temp_dir)
            
            # Process each syscall file
            for sc_file in syscall_files:
                name = sc_file.stem
                meta = metadata.get(name, {})
                
                syscalls = parse_syscall_file(sc_file)
                if len(syscalls) < 20:
                    continue
                
                if meta.get("exploit", False):
                    attack_traces.append(syscalls)
                else:
                    normal_traces.append(syscalls)
    
    logging.info(f"Found {len(normal_traces)} normal, {len(attack_traces)} attack traces")
    
    if not normal_traces and not attack_traces:
        logging.error("No valid traces found!")
        return
    
    # Split and save
    normal_split = int(0.7 * len(normal_traces))
    attack_split = int(0.7 * len(attack_traces))
    
    normal_count = process_traces(normal_traces[:normal_split], "traces_train/normal")
    attack_count = process_traces(attack_traces[:attack_split], "traces_train/attack")
    
    test_normal_count = process_traces(normal_traces[normal_split:], "traces_infer/normal")
    test_attack_count = process_traces(attack_traces[attack_split:], "traces_infer/attack")
    
    logging.info(f"Train: {normal_count} normal, {attack_count} attack")
    logging.info(f"Test: {test_normal_count} normal, {test_attack_count} attack")


def main() -> None:
    """
    Main entry point for the syscall classification pipeline.
    
    Configures logging, parses command line arguments, and routes processing
    to the appropriate handler based on the input dataset format.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    args = parse_arguments()
    
    if args.lidds:
        _process_lidds_pipeline(args)
    elif args.normal or args.attack:
        _process_h5_pipeline(args)
    else:
        logging.error("No dataset specified. Use -l for LID-DS or -n/-a for H5")


def _process_lidds_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the complete LID-DS processing pipeline.
    
    Processes the LID-DS dataset and optionally runs graph preprocessing
    if requested and the preprocessing module is available.
    
    Args:
        args: Parsed command line arguments containing dataset path and options
    """
    process_lidds_dataset(args.lidds)
    _run_preprocessing_if_requested(args)


def _process_h5_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the complete H5 format processing pipeline.
    
    Validates that both normal and attack H5 files are available,
    processes them, and optionally runs graph preprocessing.
    
    Args:
        args: Parsed command line arguments containing H5 paths and options
    """
    normal_h5 = args.normal or os.getenv("LID_NORMAL")
    attack_h5 = args.attack or os.getenv("LID_ATTACK")
    
    if not normal_h5 or not attack_h5:
        logging.error("Need both normal and attack H5 files")
        return
    
    process_h5_datasets(normal_h5, attack_h5)
    _run_preprocessing_if_requested(args)


def _run_preprocessing_if_requested(args: argparse.Namespace) -> None:
    """
    Run graph preprocessing if requested and the module is available.
    
    Checks if preprocessing was requested via command line flag and if the
    preprocessing module was successfully imported. If both conditions are met,
    runs preprocessing on the training traces.
    
    Args:
        args: Parsed command line arguments containing preprocessing flag
    """
    if args.preprocess and preprocess_dataset:
        output = preprocess_dataset("traces_train", False, False, "processed_graphs.pkl")
        logging.info(f"Preprocessing complete: {output}")


if __name__ == "__main__":
    main()