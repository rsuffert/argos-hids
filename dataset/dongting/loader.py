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
import h5py

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

def generate_h5s(baseline_df: pd.DataFrame, syscall_map: Dict[str, int], *raw_data_dirs: str) -> None:
    def find_seq_file_by_bugname(bugname: str, *dirs: str) -> Union[str, None]:
        # For some reason, sometimes the bug names in the dataframe do not include the "sy_"
        # prefix or the ".log" suffix as the raw files do, so we ensure they're added here.
        filename = bugname
        if not filename.startswith("sy_"):
            filename = "sy_" + filename
        if not filename.endswith(".log"):
            filename = filename + ".log"
        for dirname in dirs:
            for root, _, files in os.walk(dirname):
                filtered_files = list(filter(lambda f: f == filename, files))
                if not filtered_files:
                    continue
                return os.path.join(root, filtered_files[0])
        return None

    encode = lambda seq: [syscall_map[sname] for sname in seq.split("|")]

    LABEL_COL = "kcb_seq_lables"
    SPLIT_COL = "kcb_seq_class"
    BNAME_COL = "kcb_bug_name"
    MAX_SEQ_LEN = 4495

    for label in baseline_df[LABEL_COL].unique():
        for split in baseline_df[SPLIT_COL].unique():
            filtered_df = baseline_df[(baseline_df[LABEL_COL] == label) & (baseline_df[SPLIT_COL] == split)]
            h5name = f'syscall_seqs_{label}_{split}.h5'
            with h5py.File(h5name, 'w') as h5f:
                dset = h5f.create_dataset("sequences",
                    shape=(len(filtered_df), MAX_SEQ_LEN), maxshape=(None, None),
                    dtype='int16', compression='gzip'
                )
                for i, row in filtered_df.iterrows():
                    seq_file = find_seq_file_by_bugname(row[BNAME_COL], *raw_data_dirs)
                    if not seq_file:
                        print(f"Warning: Sequence file for bug name '{row[BNAME_COL]}' not found in provided directories.")
                        continue
                    with open(seq_file, 'r') as f:
                        encoded_seq = encode(f.read().strip())
                    if len(encoded_seq) > dset.shape[1]:
                        dset.resize((dset.shape[0], len(encoded_seq)))
                    # Write the encoded sequence to the dataset
                    dset[i, :len(encoded_seq)] = encoded_seq
                    # Pad the rest with -1 to indicate unused space
                    dset[i, len(encoded_seq):] = -1
            assert os.path.exists(h5name), f"Failed to create {h5name}."
            assert os.path.getsize(h5name) > 0, f"{h5name} is empty after creation."
            print(f"Successfully generated '{h5name}' with {len(filtered_df)} sequences for label '{label}' and split '{split}'.")

if __name__ == "__main__":
    syscall_map = parse_syscall_tbl(SYSCALL_TBL_PATH)
    assert syscall_map, "Syscall map is empty. Check the syscall table file or path."
    print(f"Loaded {len(syscall_map)} syscalls from the syscall table.")

    baseline_df = load_baseline_xlsx(BASELINE_XLSX_PATH)
    assert not baseline_df.empty, "Baseline DataFrame is empty. Check the .xlsx file or path."
    print(f"Loaded baseline DataFrame with {len(baseline_df)} rows.")

    print("Generating .h5 files for normal and abnormal sequences. This may take a while...")
    generate_h5s(
        baseline_df,
        syscall_map,
        os.path.expanduser(NORMAL_DATA_FOLDER_PATH),
        os.path.expanduser(ABNORMAL_DATA_FOLDER_PATH)
    )