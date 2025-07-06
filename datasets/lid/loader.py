#!/usr/bin/env python3
"""LID-DS Loader with External Syscall Table and Exploit-based Labeling."""

from pathlib import Path
import os
import logging
from typing import Tuple, List
import zipfile
import json
from datasets.dongting import parse_syscall_tbl, append_seq_to_h5

LID_DATA_DIR     = os.getenv("LID_DATA_DIR", "lid-data")
SYSCALL_TBL_PATH = os.getenv("SYSCALL_TBL_PATH",
                             os.path.join(os.path.dirname(__file__), "..", "syscall_64.tbl"))

def get_split_from_path(path: str) -> str:
    """
    Extracts the split (training, test, validation) from the file path.
    
    Args:
        path (str): The file path to extract the split from.
    
    Returns:
        str: The split type, either "training", "test", or "validation".
            If no split is found, defaults to "training".
    """
    for split in ["training", "test", "validation"]:
        if split in path:
            return split
    return "training"

def extract_label_and_seq_from_zip(path: str) -> Tuple[int, List[str]]:
    """
    Extracts the label and sequence of system calls from a zip file.

    Args:
        path (str): The path to the zip file.

    Returns:
        Tuple[int, List[str]]: A tuple containing the label (int) and a list of syscall names.
    """
    label: int = -1
    sequence: List[str] = []

    with zipfile.ZipFile(path, "r") as zf:
        # Extract the label from the .json metadata file
        metadata_files = [f for f in zf.namelist() if f.endswith(".json")]
        if not metadata_files:
            logging.warning(f"No metadata files found in zip: {path}")
            return label, sequence
        with zf.open(metadata_files[0]) as mf:
            metadata = json.load(mf)
        label = 1 if metadata["exploit"] else 0 # "exploit" key is a boolean indicating if the sample is an exploit

        # Load the syscall sequence from the .sc file
        syscall_files = [f for f in zf.namelist() if f.endswith(".sc")]
        if not syscall_files:
            logging.warning(f"No syscall files found in zip: {path}")
            return label, sequence
        with zf.open(syscall_files[0]) as sf:
            lines = sf.readlines()
        sequence = [line.split()[5].decode("utf-8") # the syscall ID is the 6th element in each line
                    for line in lines]
        
    return label, sequence

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse the syscall table to map syscall names to their numeric IDs for ML model training
    syscalls_map = parse_syscall_tbl(SYSCALL_TBL_PATH)
    assert syscalls_map, f"Failed to parse syscall table from '{SYSCALL_TBL_PATH}'"
    
    syscall_dirs_iterator = Path(LID_DATA_DIR).rglob("*.zip") # Lists ALL ZIP files under LID_DATA_DIR
    for syscall_dir in syscall_dirs_iterator:
        syscall_dir_path = str(syscall_dir)
        
        if "__MACOSX" in syscall_dir_path:
            continue
        logging.debug(f"Processing syscall directory: {syscall_dir_path}")

        split = get_split_from_path(syscall_dir_path)
        logging.debug(f"Using split '{split}'")

        label, sequence = extract_label_and_seq_from_zip(syscall_dir_path)
        sequence_ids = [syscalls_map[syscall_name] for syscall_name in sequence]
        logging.debug(f"Using label '{label}' with sequence of length {len(sequence_ids)}")

        append_seq_to_h5(
            sequence_ids,
            os.path.join(LID_DATA_DIR, f"{label}_{split}.h5")
        )