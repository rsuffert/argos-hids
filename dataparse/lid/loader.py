#!/usr/bin/env python3
"""LID-DS Loader with Dynamic Syscall Table Generation and Exploit Labeling."""

from pathlib import Path
import os
import logging
import pickle
from typing import Tuple, List, Dict, Optional
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

def extract_metadata_from_zip(zf: zipfile.ZipFile, path: str) -> Tuple[int, Optional[int]]:
    """
    Extract label and attack timestamp from ZIP metadata.
    
    Args:
        zf: Open ZipFile object
        path: ZIP file path for logging
        
    Returns:
        Tuple[int, Optional[int]]: (label, attack_timestamp)
    """
    metadata_files = [f for f in zf.namelist() if f.endswith(".json")]
    if not metadata_files:
        logging.warning(f"No metadata files found in zip: {path}")
        return -1, None
        
    with zf.open(metadata_files[0]) as mf:
        metadata = json.load(mf)
        
    label = 1 if metadata.get("exploit") else 0
    attack_ts = None
    
    if label:
        exploit_info = metadata.get("time", {}).get("exploit", [])
        if exploit_info and "absolute" in exploit_info[0]:
            attack_ts = int(str(exploit_info[0]["absolute"]).split(".")[0])
            
    return label, attack_ts

def parse_syscall_file(zf: zipfile.ZipFile, path: str) -> List[Tuple[int, str]]:
    """
    Parse syscall file and return list of (timestamp, syscall_name) tuples.
    
    Args:
        zf: Open ZipFile object
        path: ZIP file path for logging
        
    Returns:
        List[Tuple[int, str]]: List of (timestamp, syscall_name) pairs
    """
    syscall_files = [f for f in zf.namelist() if f.endswith(".sc")]
    if not syscall_files:
        logging.warning(f"No syscall files found in zip: {path}")
        return []
        
    with zf.open(syscall_files[0]) as sf:
        lines = sf.readlines()
        
    parsed = []
    for line in lines:
        try:
            timestamp = int(str(line.split()[0].decode()).split(".")[0])
            syscall_name = line.split()[5].decode()
            parsed.append((timestamp, syscall_name))
        except (IndexError, UnicodeDecodeError):
            continue
            
    return parsed

def select_window_sequence(
    parsed: List[Tuple[int, str]], label: int, attack_ts: Optional[int]
) -> List[Tuple[int, str]]:
    """
    Select appropriate window sequence based on label and attack timestamp.
    
    Args:
        parsed: List of (timestamp, syscall_name) pairs
        label: 1 for attack, 0 for normal
        attack_ts: Attack timestamp (None if not available)
        
    Returns:
        List[Tuple[int, str]]: Selected window sequence
    """
    if label and attack_ts:
        for idx, (ts, _name) in enumerate(parsed):
            if ts >= attack_ts:
                return parsed[idx:idx+WINDOW_SIZE]
        return []
    return parsed[:WINDOW_SIZE]

def convert_to_syscall_ids(window_seq: List[Tuple[int, str]], syscall_dict: Dict[str, int]) -> List[int]:
    """
    Convert syscall names to IDs using dictionary.
    
    Args:
        window_seq: List of (timestamp, syscall_name) pairs
        syscall_dict: Dictionary mapping syscall names to IDs
        
    Returns:
        List[int]: List of syscall IDs
    """
    return [get_or_add_syscall_id(name, syscall_dict) for _, name in window_seq]

def extract_label_and_seq_from_zip(path: str, syscall_dict: Dict[str, int]) -> Tuple[int, List[int]]:
    """
    Extracts the label and sequence of system calls from a zip file.

    Args:
        path (str): The path to the zip file.
        syscall_dict (Dict[str, int]): Dictionary mapping syscall names to IDs.

    Returns:
        Tuple[int, List[int]]: A tuple containing the label (int) and a list of syscall IDs.
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Extract metadata
            label, attack_ts = extract_metadata_from_zip(zf, path)
            if label == -1:
                return label, []
                
            # Parse syscall file
            parsed = parse_syscall_file(zf, path)
            if not parsed:
                return label, []
                
            # Select window sequence
            window_seq = select_window_sequence(parsed, label, attack_ts)
            
            # Convert to syscall IDs
            window_ids = convert_to_syscall_ids(window_seq, syscall_dict)
            
        return label, window_ids
        
    except Exception as e:
        logging.error(f"Error processing {path}: {e}")
        return -1, []

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