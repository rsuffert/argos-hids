#!/usr/bin/env python3
"""LID-DS Loader with Dynamic Syscall Table Generation and Exploit Labeling."""

from pathlib import Path
import os
import logging
import pickle
from typing import Tuple, List, Dict
import zipfile
import json
from dataparse.dongting import append_seq_to_h5

# Configuration
WINDOW_SIZE = 512              # Maximum number of syscalls to extract per window(in attack data)
# Path to the directory containing LID dataset and where processed .h5 files will be stored
LID_DATA_DIR = os.getenv("LID_DATA_DIR", "lid-data")
SYSCALL_DICT_PATH = os.path.join(LID_DATA_DIR, "syscall_dict.pkl") # Path to where the syscall dictionary file is saved

def load_or_create_syscall_dict() -> Dict[str, int]:
    """
    Load existing syscall dictionary or create a new one.
    
    Returns:
        Dict[str, int]: Dictionary mapping syscall names to IDs.
    """
    if os.path.exists(SYSCALL_DICT_PATH):
        with open(SYSCALL_DICT_PATH, "rb") as f:
            syscall_dict = pickle.load(f)
        logging.info(f"Loaded syscall dictionary with {len(syscall_dict)} entries")
    else:
        syscall_dict = {}
        logging.info("Created new syscall dictionary")
    return syscall_dict

def save_syscall_dict(syscall_dict: Dict[str, int]) -> None:
    """
    Save syscall dictionary to disk.
    
    Args:
        syscall_dict (Dict[str, int]): Dictionary to save.
    """
    with open(SYSCALL_DICT_PATH, "wb") as f:
        pickle.dump(syscall_dict, f)
    logging.info(f"Saved syscall dictionary with {len(syscall_dict)} entries")

def get_or_add_syscall_id(syscall_name: str, syscall_dict: Dict[str, int]) -> int:
    """
    Get existing ID for syscall or add new one if not present.
    
    Args:
        syscall_name (str): Name of the syscall.
        syscall_dict (Dict[str, int]): Current syscall dictionary.
    
    Returns:
        int: ID for the syscall.
    """
    if syscall_name not in syscall_dict:
        # Assign next available ID
        new_id = len(syscall_dict)
        syscall_dict[syscall_name] = new_id
        logging.debug(f"Added new syscall '{syscall_name}' with ID {new_id}")
    return syscall_dict[syscall_name]

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

def extract_label_and_seq_from_zip(path: str, syscall_dict: Dict[str, int]) -> Tuple[int, List[int]]:
    """
    Extracts the label and sequence of system calls from a zip file.

    Args:
        path (str): The path to the zip file.
        syscall_dict (Dict[str, int]): Dictionary mapping syscall names to IDs.

    Returns:
        Tuple[int, List[int]]: A tuple containing the label (int) and a list of syscall IDs.
    """
    label: int = -1
    window_ids: List[int] = []

    with zipfile.ZipFile(path, "r") as zf:
        # Extract the label and exploit timestamp from the .json metadata file
        metadata_files = [f for f in zf.namelist() if f.endswith(".json")]
        if not metadata_files:
            logging.warning(f"No metadata files found in zip: {path}")
            return label, []
        with zf.open(metadata_files[0]) as mf:
            metadata = json.load(mf)
        label = 1 if metadata.get("exploit") else 0
        attack_ts = None
        exploit_info = metadata.get("time", {}).get("exploit", [])
        if label and exploit_info and "absolute" in exploit_info[0]:
            attack_ts = int(str(exploit_info[0]["absolute"]).split(".")[0])

        # Load and parse the syscall sequence from the .sc file
        syscall_files = [f for f in zf.namelist() if f.endswith(".sc")]
        if not syscall_files:
            logging.warning(f"No syscall files found in zip: {path}")
            return label, []
        with zf.open(syscall_files[0]) as sf:
            lines = sf.readlines()
        # Parse: (timestamp, syscall_name)
        parsed = [(int(str(line.split()[0].decode()).split(".")[0]), line.split()[5].decode()) for line in lines]

        # Select window starting at attack timestamp
        if label and attack_ts:
            for idx, (ts, _name) in enumerate(parsed):
                if ts >= attack_ts:
                    window_seq = parsed[idx:idx+WINDOW_SIZE]
                    break
            else:
                window_seq = []
        else:
            window_seq = parsed[:WINDOW_SIZE]

        # Convert syscall names to IDs, creating new IDs as needed
        window_ids = [get_or_add_syscall_id(name, syscall_dict) for _, name in window_seq]
    
    return label, window_ids

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load or create the dynamic syscall dictionary
    syscall_dict = load_or_create_syscall_dict()
    
    syscall_dirs_iterator = Path(LID_DATA_DIR).rglob("*.zip") # Lists ALL ZIP files under LID_DATA_DIR
    for syscall_dir in syscall_dirs_iterator:
        syscall_dir_path = str(syscall_dir)
        
        if "__MACOSX" in syscall_dir_path:
            continue
        logging.debug(f"Processing syscall directory: {syscall_dir_path}")

        split = get_split_from_path(syscall_dir_path)
        logging.debug(f"Using split '{split}'")

        label, window_ids = extract_label_and_seq_from_zip(syscall_dir_path, syscall_dict)
        if window_ids:
            logging.debug(f"Using label '{label}' with sequence of length {len(window_ids)}")
            append_seq_to_h5(
                window_ids,
                os.path.join(LID_DATA_DIR, f"{label}_{split}.h5")
            )
    
    # Save the updated syscall dictionary
    save_syscall_dict(syscall_dict)
    logging.info(f"Processing complete. Syscall dictionary has {len(syscall_dict)} entries.")