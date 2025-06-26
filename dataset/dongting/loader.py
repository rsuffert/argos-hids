"""
loader.py

This script loads system call sequences for the DongTing dataset from a labeled baseline Excel file
and corresponding log files(stored in normal and abnormal folders), encodes them using a syscall table,
and exports the processed sequences into compressed NumPy .npz files for each label (normal/abnormal).
All of these files are provided as part of the DongTing dataset and can be downloaded from here:
https://zenodo.org/records/6627050.

The script supports recursive search for log files and can be used for preparing datasets for anomaly
detection or intrusion detection experiments.

Usage:
    python loader.py

Environment Variables:
    BASELINE_XLSX_PATH        Path to the baseline DongTing Excel file.
    NORMAL_DATA_FOLDER_PATH   Path to the DongTing folder containing normal log files.
    ABNORMAL_DATA_FOLDER_PATH Path to the DongTing folder containing abnormal log files.

Outputs:
    syscall_seqs_label_0.npz  Compressed file with DongTing normal sequences, categorized in training, testing, and validation splits.
    syscall_seqs_label_1.npz  Compressed file with DongTing abnormal sequences, categorized in training, testing, and validation splits.
"""

from typing import Dict, List, Union
import os
import pandas as pd
import numpy as np

def get_mandatory_env_var(name: str) -> str:
	value = os.getenv(name)
	if not value:
		raise EnvironmentError(f"Environment variable '{name}' is required but not set.")
	return value

BASELINE_XLSX_PATH: str        = get_mandatory_env_var('BASELINE_XLSX_PATH')
NORMAL_DATA_FOLDER_PATH: str   = get_mandatory_env_var('NORMAL_DATA_FOLDER_PATH')
ABNORMAL_DATA_FOLDER_PATH: str = get_mandatory_env_var('ABNORMAL_DATA_FOLDER_PATH')
SYSCALL_TBL_PATH: str          = os.path.join(os.path.dirname(__file__), 'syscall_64.tbl')

def parse_syscall_tbl(path: str) -> Dict[str, int]:
    """
    Parses a system call table file and returns a mapping from syscall names to their IDs.

    Args:
        path (str): The path to the system call table file. Each line in the file should contain at least three fields:
            - The first field is the syscall ID (integer).
            - The third field is the syscall name (string).
            Lines starting with '#' or empty lines are ignored.

    Returns:
        Dict[str, int]: A dictionary mapping syscall names to their corresponding IDs.
    """
    syscalls_map = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            syscall_id = int(parts[0])
            syscall_name = parts[2]
            syscalls_map[syscall_name] = syscall_id
    return syscalls_map

def load_baseline_xlsx(path: str) -> pd.DataFrame:
    """
    Loads an Excel (.xlsx) file from the specified path into a pandas DataFrame.

    Args:
        path (str): The file path to the Excel file. The path can include '~' to represent the user's home directory.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the Excel file.
    """
    return pd.read_excel(os.path.expanduser(path))

def load_labeled_syscall_sequences(baseline_df: pd.DataFrame, syscall_map: Dict[str, int], normal_folder: str, abnormal_folder: str) -> Dict[int, Dict[str, List[List[int]]]]:
    """
    Loads and encodes labeled system call sequences from files specified in a baseline DataFrame,
    organizing them by label and dataset split.

    This function iterates over each entry in the provided DataFrame, constructs the corresponding filename,
    and searches for the file in the specified normal and abnormal folders. If found, it reads the system call sequence,
    encodes it into a list of integer IDs using the provided syscall map, and stores it in a nested dictionary structure
    organized by label (normal or abnormal) and dataset split (train, test, validation).

    Args:
        baseline_df (pd.DataFrame): DataFrame containing metadata for each sequence, including bug name, label, and class (split).
        syscall_map (Dict[str, int]): Mapping from syscall names to integer IDs.
        normal_folder (str): Path to the folder containing normal sequences.
        abnormal_folder (str): Path to the folder containing abnormal sequences.

    Returns:
        Dict[int, Dict[str, List[List[int]]]]: Nested dictionary where the first key is the label (0 for normal, 1 for abnormal), the second key is the dataset split ("DTDS-train", "DTDS-test", "DTDS-validation"), and the value is a list of encoded syscall sequences (each sequence is a list of integer IDs).

    Raises:
        KeyError: If a syscall in a sequence is not found in the syscall_map.

    Notes:
        - If a file corresponding to a DataFrame entry is not found in either folder, a warning is printed and the entry is skipped.
        - The function assumes that the bug names in the DataFrame may need to be prefixed with "sy_" and suffixed with ".log" to match the actual filenames.
    """
    def encode_syscall_sequence(sequence: str, syscall_map: Dict[str, int]) -> List[int]:
        """
        Encodes a sequence of system calls into a list of integer IDs based on a provided mapping.

        Args:
            sequence (str): A string of system calls separated by the '|' character.
            syscall_map (Dict[str, int]): A dictionary mapping syscall names to integer IDs.

        Returns:
            List[int]: A list of integer IDs corresponding to the system calls in the input sequence.

        Raises:
            KeyError: If a syscall in the sequence is not found in the syscall_map.
        """
        encoded_sequence = []
        for sc in sequence.split('|'):
            if sc not in syscall_map:
                raise KeyError(f"Syscall '{sc}' not found in syscall map.")
            encoded_sequence.append(syscall_map[sc])
        return encoded_sequence
    def find_file_in_folder(filename: str, folder: str) -> Union[str, None]:
        """
        Searches for a file with the specified filename within the given folder (recursively).
        If found, reads and returns the contents of the file as a stripped string.
        If the file is not found, returns None.

        Args:
            filename (str): The name of the file to search for.
            folder (str): The path to the folder in which to search.

        Returns:
            Union[str, None]: The stripped contents of the found file, or None if not found.
        """
        for root, _, files in os.walk(os.path.expanduser(folder)):
            for fname in files:
                if fname != filename:
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, 'r') as f:
                    return f.read().strip()
        return None

    # Skeleton for the dictionary to be returned.
    # Each list will contain a List[int], which represents a syscall sequence where
    # each int is representing a syscall ID as per the syscall map.
    sequences_per_label = {
        1: {
            "DTDS-train": [],
            "DTDS-test": [],
            "DTDS-validation": []
        },
        0: {
            "DTDS-train": [],
            "DTDS-test": [],
            "DTDS-validation": []
        }
    }
    for idx, row in baseline_df.iterrows():
        # For some reason the bug names in the dataset do not include the "sy_" prefix or ".log" extension
        # as their correspoding files in the dataset, so we ensure they're added here to locate the files
        # correctly.
        fname = row["kcb_bug_name"]
        if not fname.startswith("sy_"):
            fname = "sy_" + fname
        if not fname.endswith(".log"):
            fname = fname + ".log"
        label = 0 if row["kcb_seq_lables"] == "Normal" else 1
        _class = row["kcb_seq_class"]
        content = find_file_in_folder(fname, normal_folder) or find_file_in_folder(fname, abnormal_folder)
        if not content:
            print(f"Warning: File '{fname}' not found in the dataset. Skipping this entry.")
            continue
        encoded_sequence = encode_syscall_sequence(content, syscall_map)
        sequences_per_label[label][_class].append(encoded_sequence)
        print(f"({idx+1}) Successfully loaded sequence for file '{fname}' with label {label}.")
    return sequences_per_label

def store_labeled_sequences_to_npz(sequences: Dict[int, Dict[str, List[List[int]]]]) -> None:
    """
    Stores labeled syscall sequences into compressed NumPy .npz files.

    For each label in the input dictionary, this function saves the corresponding
    training, testing, and validation sequences into a separate compressed .npz file.
    Each file is named 'syscall_seqs_label_{label}.npz', where {label} is the label key.

    Args:
        sequences (Dict[int, Dict[str, List[List[int]]]]): 
            A dictionary mapping integer labels to another dictionary with keys 
            'DTDS-train', 'DTDS-test', and 'DTDS-validation', each containing a list 
            of syscall sequences (lists of integers).

    Returns:
        None

    Side Effects:
        Writes compressed .npz files to disk for each label in the input dictionary.
        Prints a message for each file stored.
    """
    for label, data in sequences.items():
        npz_filename = f"syscall_seqs_label_{label}.npz"
        np.savez_compressed(
            npz_filename,
            train=np.array(sequences[label]["DTDS-train"], dtype=object),
            test=np.array(sequences[label]["DTDS-test"], dtype=object),
            validation=np.array(sequences[label]["DTDS-validation"], dtype=object),
        )
        print(f"Stored labeled sequences for label {label} in '{npz_filename}' (compressed).")

if __name__ == "__main__":
    syscall_map = parse_syscall_tbl(SYSCALL_TBL_PATH)
    assert syscall_map, "Syscall map is empty. Check the syscall table file or path."
    print(f"Loaded {len(syscall_map)} syscalls from the syscall table.")

    baseline_df = load_baseline_xlsx(BASELINE_XLSX_PATH)
    assert not baseline_df.empty, "Baseline DataFrame is empty. Check the .xlsx file or path."
    print(f"Loaded baseline DataFrame with {len(baseline_df)} rows.")

    print(f"Loading labeled syscall sequences from '{NORMAL_DATA_FOLDER_PATH}' and '{ABNORMAL_DATA_FOLDER_PATH}' (this can take a while)...")
    labeled_sequences = load_labeled_syscall_sequences(
        baseline_df, syscall_map, NORMAL_DATA_FOLDER_PATH, ABNORMAL_DATA_FOLDER_PATH
    )
    assert labeled_sequences[0] and labeled_sequences[1], "Labeled syscall sequences were not loaded correctly. Check the data files and paths."
    print(f"Successfully loaded {len(labeled_sequences[0])} normal and {len(labeled_sequences[1])} abnormal syscall sequences.")

    print("Storing labeled syscall sequences to .npz files...")
    store_labeled_sequences_to_npz(labeled_sequences)
