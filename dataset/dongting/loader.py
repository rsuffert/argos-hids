from typing import Dict, List, Union, Callable
import os
import numpy as np
import h5py
import concurrent.futures
import logging
import pandas as pd

SYSCALL_TBL_PATH          = os.getenv('SYSCALL_TBL_PATH',          os.path.join(os.path.dirname(__file__), 'syscall_64.tbl'))
NORMAL_DATA_FOLDER_PATH   = os.getenv('NORMAL_DATA_FOLDER_PATH',   os.path.join(os.path.dirname(__file__), 'Normal_data'))
ABNORMAL_DATA_FOLDER_PATH = os.getenv('ABNORMAL_DATA_FOLDER_PATH', os.path.join(os.path.dirname(__file__), 'Abnormal_data'))
BASELINE_XLSX_PATH        = os.getenv('BASELINE_XLSX_PATH',        os.path.join(os.path.dirname(__file__), 'Baseline.xlsx'))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

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

def parse_raw_seq_file(path: str, separator: str, syscall_map: Dict[str, int]) -> List[int]:
    """
    Parses a raw sequence file containing system call names separated by a specified separator,
    and maps each system call name to its corresponding integer ID using the provided syscall_map.

    Args:
        path (str): The path to the raw sequence file.
        separator (str): The string used to separate system call names in the file.
        syscall_map (Dict[str, int]): A dictionary mapping system call names to integer IDs.

    Returns:
        List[int]: A list of integer IDs corresponding to the system call names in the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If a system call name in the file is not found in syscall_map.
    """
    with open(path, 'r') as f:
        content = f.read().strip()
    return list(map(
        lambda sysname: int(syscall_map[sysname]),
        content.split(separator)
    ))

def append_seq_to_h5(sequence: List[int], h5_path: str) -> None:
    """
    Appends a sequence of integers to an HDF5 file under the dataset "sequences".

    If the "sequences" dataset does not exist, it is created as a variable-length dataset
    of 16-bit integers with gzip compression. Each call appends the given sequence as a new entry.

    Args:
        sequence (List[int]): The sequence of integers to append.
        h5_path (str): Path to the HDF5 file.
    """
    arr = np.array(sequence, dtype=np.int16)
    with h5py.File(h5_path, 'a') as h5f:
        if "sequences" not in h5f:
            h5f.create_dataset("sequences",
                shape=(0,), maxshape=(None,),
                dtype=h5py.special_dtype(vlen=np.dtype('int16')),
                compression="gzip"
            )
        dset = h5f["sequences"]
        dset.resize((dset.shape[0] + 1,))
        dset[-1] = arr

def parse_and_store_sequences(
        *base_dirs: str,
        file_parser: Callable[[str], List[int]],
        label_and_class_getter: Callable[str, Union[None, tuple[int, str]]],
        trim_log_ext: bool = True
    ) -> None:
    """
    Parses files from given base directories, extracts sequences using a file parser, retrieves labels and classes, and stores the sequences in HDF5 files.
    Args:
        *base_dirs (str): One or more base directory paths to search for files.
        file_parser (Callable[[str], List[int]]): Function that takes a file path and returns a sequence as a list of integers.
        label_and_class_getter (Callable[[str], Union[None, tuple[int, str]]]): Function that takes a bug name and returns a tuple (label, class) or None if not found.
        trim_log_ext (bool, optional): Whether to trim the ".log" extension from file names when extracting the bug name. Defaults to True.
    Side Effects:
        - Appends parsed sequences to HDF5 files named "{label}_{class_}.h5" in the current working directory.
    """
    for basedir_path in base_dirs:
        for root, _, files in os.walk(basedir_path):
            for fname in files:
                if fname.endswith(".h5"): continue # Skip the H5 files themselves
                
                fpath = os.path.join(root, fname)

                bugname = fname
                if bugname.startswith("sy_"):                 bugname = bugname[3:]
                if trim_log_ext and bugname.endswith(".log"): bugname = bugname[:-4]
                
                logging.debug(f"Processing file: {fpath}")
                sequence = file_parser(fpath)
                label, class_ = label_and_class_getter(bugname)
                if not label or not class_:
                    logging.warning(f"Label or class not found for bug name '{bugname}' in file '{fpath}'. Skipping...")
                    continue
                
                append_seq_to_h5(sequence, f"{label}_{class_}.h5")

if __name__ == "__main__":
    assert os.path.exists(SYSCALL_TBL_PATH), f"Syscall table file not found: {SYSCALL_TBL_PATH}"
    syscall_map = parse_syscall_tbl(SYSCALL_TBL_PATH)
    assert syscall_map, "Syscall map is empty. Check the syscall table file or path."
    logging.info(f"Loaded {len(syscall_map)} syscalls from the syscall table.")

    assert os.path.exists(BASELINE_XLSX_PATH), f"Baseline file not found: {BASELINE_XLSX_PATH}"
    baseline_df = pd.read_excel(BASELINE_XLSX_PATH)
    assert not baseline_df.empty, f"Baseline DataFrame is empty. Check the file: {BASELINE_XLSX_PATH}"
    logging.info(f"Loaded baseline data with {len(baseline_df)} rows from.")

    file_parser = lambda path: parse_raw_seq_file(path, "|", syscall_map)
    def label_and_class_getter(bugname: str) -> Union[None, tuple[int, str]]:
        row = baseline_df[baseline_df["kcb_bug_name"] == bugname]
        if row.empty: return None, None
        return row["kcb_seq_lables"].values[0], row["kcb_seq_class"].values[0]

    assert os.path.exists(NORMAL_DATA_FOLDER_PATH), f"Normal data folder not found: {NORMAL_DATA_FOLDER_PATH}"
    assert os.path.isdir(NORMAL_DATA_FOLDER_PATH), f"Path is not a directory: {NORMAL_DATA_FOLDER_PATH}"
    logging.info(f"Starting to parse sequences from '{NORMAL_DATA_FOLDER_PATH}'")
    parse_and_store_sequences(
        NORMAL_DATA_FOLDER_PATH,
        file_parser=file_parser,
        label_and_class_getter=label_and_class_getter,
        trim_log_ext=False
    )

    assert os.path.exists(ABNORMAL_DATA_FOLDER_PATH), f"Abnormal data folder not found: {ABNORMAL_DATA_FOLDER_PATH}"
    assert os.path.isdir(ABNORMAL_DATA_FOLDER_PATH), f"Path is not a directory: {ABNORMAL_DATA_FOLDER_PATH}"
    logging.info(f"Starting to parse sequences from '{ABNORMAL_DATA_FOLDER_PATH}'")
    parse_and_store_sequences(
        ABNORMAL_DATA_FOLDER_PATH,
        file_parser=file_parser,
        label_and_class_getter=label_and_class_getter
    )