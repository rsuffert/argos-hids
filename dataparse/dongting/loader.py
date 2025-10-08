"""
Module for parsing system call sequences from DongTing dataset raw files and storing them
in HDF5 compressed and pre-processed format for seamless integration with machine learning
workflows.
"""

from typing import Dict, List, Union, Callable, Tuple
import os
import logging
import numpy as np
import h5py
import pandas as pd

SYSCALL_TBL_PATH          = os.getenv("SYSCALL_TBL_PATH",
                                      os.path.join(os.path.dirname(__file__), "syscall_64.tbl"))
NORMAL_DATA_FOLDER_PATH   = os.getenv("NORMAL_DATA_FOLDER_PATH",
                                      os.path.join(os.path.dirname(__file__), "Normal_data"))
ABNORMAL_DATA_FOLDER_PATH = os.getenv("ABNORMAL_DATA_FOLDER_PATH",
                                      os.path.join(os.path.dirname(__file__), "Abnormal_data"))
BASELINE_XLSX_PATH        = os.getenv("BASELINE_XLSX_PATH",
                                      os.path.join(os.path.dirname(__file__), "Baseline.xlsx"))
SYSCALL_MAPPING_DUMP_PATH = "mapping.csv"

def parse_syscall_tbl(path: str) -> Dict[str, int]:
    """
    Parses a system call table file and returns a mapping from syscall names to their IDs.

    Args:
        path (str): The path to the system call table file. Each line in the file should contain
            at least three fields:
            - The first field is the syscall ID (integer).
            - The third field is the syscall name (string).
            Lines starting with "#" or empty lines are ignored.

    Returns:
        Dict[str, int]: A dictionary mapping syscall names to their corresponding IDs.
    """
    syscalls_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            syscall_id = int(parts[0])
            syscall_name = parts[2]
            syscalls_map[syscall_name] = syscall_id
    return syscalls_map

def parse_raw_seq_file(path: str, separator: str, syscall_lookup: Dict[str, int]) -> List[int]:
    """
    Parses a raw sequence file containing system call names separated by a specified separator,
    and maps each system call name to its corresponding integer ID using the provided
    syscall_lookup.

    Args:
        path (str): The path to the raw sequence file.
        separator (str): The string used to separate system call names in the file.
        syscall_lookup (Dict[str, int]): A dictionary mapping system call names to integer IDs.

    Returns:
        List[int]: A list of integer IDs corresponding to the system call names in the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If a system call name in the file is not found in syscall_lookup.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return list(map(
        lambda sysname: int(syscall_lookup[sysname]),
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
    with h5py.File(h5_path, "a") as h5f:
        if "sequences" not in h5f:
            h5f.create_dataset("sequences",
                shape=(0,), maxshape=(None,),
                dtype=h5py.special_dtype(vlen=np.dtype("int16")),
                compression="gzip"
            )
        dset = h5f["sequences"]
        dset.resize((dset.shape[0] + 1,))
        dset[-1] = arr

def parse_and_store_sequences(
        *base_dirs: str,
        file_parser: Callable[[str], List[int]],
        label_and_class_getter: Callable[[str], Union[Tuple[None, None], Tuple[int, str]]],
        trim_log_ext: bool = True
    ) -> None:
    """
    Parses files from given base directories, extracts sequences using a file parser, retrieves
        labels and classes, and stores the sequences in HDF5 files.
    Args:
        *base_dirs (str): One or more base directory paths to search for files.
        file_parser (Callable[[str], List[int]]): Function that takes a file path and returns
            a sequence as a list of integers.
        label_and_class_getter (Callable[[str], Union[None, Tuple[int, str]]]): Function that
            takes a bug name and returns a Tuple (label, class) or None if not found.
        trim_log_ext (bool, optional): Whether to trim the ".log" extension from file names when
            extracting the bug name. Defaults to True.
    Side Effects:
        - Appends parsed sequences to HDF5 files named "{label}_{class_}.h5" in the current
            working directory.
    """
    for basedir_path in base_dirs:
        for root, _, files in os.walk(basedir_path):
            for fname in files:
                if fname.endswith(".h5"):
                    continue # Skip the H5 files themselves

                fpath = os.path.join(root, fname)

                bugname = fname
                if bugname.startswith("sy_"):
                    bugname = bugname[3:]
                if trim_log_ext and bugname.endswith(".log"):
                    bugname = bugname[:-4]

                logging.debug("Processing file: %s", fpath)
                sequence = file_parser(fpath)
                label, class_ = label_and_class_getter(bugname)
                if label is None or class_ is None:
                    logging.warning("Label/class not found for '%s' in '%s'.", bugname, fpath)
                    continue

                append_seq_to_h5(sequence, f"{label}_{class_}.h5")

def main() -> None:
    """Main function to pre-process the DongTing dataset."""
    assert os.path.exists(SYSCALL_TBL_PATH), f"Syscall table file not found: {SYSCALL_TBL_PATH}"
    syscall_map = parse_syscall_tbl(SYSCALL_TBL_PATH)
    assert syscall_map, "Syscall map is empty. Check the syscall table file or path."
    logging.info("Loaded %d syscalls from the syscall table.", len(syscall_map))

    with open(SYSCALL_MAPPING_DUMP_PATH, "w") as f:
        for k, v in syscall_map.items():
            f.write(f"{k},{v}\n")
    logging.info(f"Dumping loaded syscalls to {SYSCALL_MAPPING_DUMP_PATH}")

    assert os.path.exists(BASELINE_XLSX_PATH), f"Baseline file not found: {BASELINE_XLSX_PATH}"
    baseline_df = pd.read_excel(BASELINE_XLSX_PATH)
    assert not baseline_df.empty, f"Baseline DataFrame is empty. Check: {BASELINE_XLSX_PATH}"
    logging.info("Loaded baseline data with %d rows from.", len(baseline_df))

    def raw_seq_file_closure(path: str) -> List[int]:
        """Closure function to parse a raw sequence file using the syscall map."""
        return parse_raw_seq_file(path, "|", syscall_map)

    def label_and_class_getter_closure(bugname: str) -> Union[Tuple[None, None], Tuple[int, str]]:
        """
        Closure function to retrieve the label and class for a given bug name from the
        baseline DataFrame.
        """
        row = baseline_df[baseline_df["kcb_bug_name"] == bugname]
        if row.empty:
            return None, None
        return row["kcb_seq_lables"].values[0], row["kcb_seq_class"].values[0]

    assert os.path.exists(NORMAL_DATA_FOLDER_PATH), f"{NORMAL_DATA_FOLDER_PATH} not found"
    assert os.path.isdir(NORMAL_DATA_FOLDER_PATH), f"{NORMAL_DATA_FOLDER_PATH} not a directory"
    logging.info("Starting to parse sequences from '%s'", NORMAL_DATA_FOLDER_PATH)
    parse_and_store_sequences(
        NORMAL_DATA_FOLDER_PATH,
        file_parser=raw_seq_file_closure,
        label_and_class_getter=label_and_class_getter_closure,
        trim_log_ext=False
    )

    assert os.path.exists(ABNORMAL_DATA_FOLDER_PATH), f"{ABNORMAL_DATA_FOLDER_PATH} not found"
    assert os.path.isdir(ABNORMAL_DATA_FOLDER_PATH), f"{ABNORMAL_DATA_FOLDER_PATH} not a directory"
    logging.info("Starting to parse sequences from '%s'", ABNORMAL_DATA_FOLDER_PATH)
    parse_and_store_sequences(
        ABNORMAL_DATA_FOLDER_PATH,
        file_parser=raw_seq_file_closure,
        label_and_class_getter=label_and_class_getter_closure
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()