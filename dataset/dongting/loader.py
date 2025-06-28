from typing import Dict, List, Union, Callable
import os
import numpy as np
import h5py
import concurrent.futures

def get_mandatory_env_var(name: str) -> str:
	value = os.getenv(name)
	if not value:
		raise EnvironmentError(f"Environment variable '{name}' is required but not set.")
	return value

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
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Path is not a file: {path}")
    if not path.endswith(".log"):
        raise ValueError(f"File {path} should be a .log file from the DongTing dataset")
    with open(path, 'r') as f:
        content = f.read().strip()
    return list(map(
        lambda sysname: int(syscall_map[sysname]),
        content.split(separator)
    ))

def process_and_store_sequences(
    base_dir_path: str,
    file_parser: Callable[str, List[int]],
    batch_size: int = 1000,
    ) -> None:
    """
    Processes sequence files in a directory, parses them using a provided parser, and stores the resulting sequences in an HDF5 file.
    Args:
        base_dir_path (str): Path to the directory containing sequence files to process.
        file_parser (Callable[[str], List[int]]): Function that takes a file path and returns a list of integers representing a sequence.
        batch_size (int, optional): Number of sequences to process and write to the HDF5 file at a time. Defaults to 1000.
    Creates:
        An HDF5 file named 'sequences.h5' in the specified directory, containing all parsed sequences as variable-length arrays of int16, compressed with gzip.
    """
    if not os.path.exists(base_dir_path):
        raise FileNotFoundError(f"Directory not found: {base_dir_path}")
    if not os.path.isdir(base_dir_path):
        raise NotADirectoryError(f"Path is not a directory: {base_dir_path}")
    
    h5py_path = os.path.join(base_dir_path, 'sequences.h5')
    with h5py.File(h5py_path, 'w') as h5f:
        dset = h5f.create_dataset("sequences",
            shape=(0,), maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype('int16')),
            compression="gzip"
        )

        batch = []
        for fname in os.listdir(base_dir_path):
            if fname.endswith('.h5'): continue # Skip the HDF5 file itself
            fpath = os.path.join(base_dir_path, fname)
            print(f"Processing sequences from file: {fpath}")
            seq = file_parser(fpath)
            batch.append(np.array(seq, dtype=np.int16))
            if len(batch) == batch_size:
                dset.resize((dset.shape[0] + batch_size,))
                dset[-batch_size:] = batch
                batch = []
        
        # Write any remaining sequences in the batch
        if batch:
            dset.resize((dset.shape[0] + len(batch),))
            dset[-len(batch):] = batch

if __name__ == "__main__":
    assert os.path.exists(SYSCALL_TBL_PATH), f"Syscall table file does not exist: {SYSCALL_TBL_PATH}"
    assert os.path.isfile(SYSCALL_TBL_PATH), f"Syscall table path is not a file: {SYSCALL_TBL_PATH}"

    syscall_map = parse_syscall_tbl(SYSCALL_TBL_PATH)
    assert syscall_map, "Syscall map is empty. Check the syscall table file or path."
    print(f"Loaded {len(syscall_map)} syscalls from the syscall table.")

    file_parser = lambda path: parse_raw_seq_file(path, "|", syscall_map)

    subdirs = []

    expanded_normal_path = os.path.expanduser(NORMAL_DATA_FOLDER_PATH)
    for dirname1 in os.listdir(expanded_normal_path):
        dirpath1 = os.path.join(expanded_normal_path, dirname1)
        if not os.path.isdir(dirpath1): continue
        for dirname2 in os.listdir(dirpath1):
            dirpath2 = os.path.join(dirpath1, dirname2)
            if not os.path.isdir(dirpath2): continue
            print(f"Processing normal sequences from {dirpath2}")
            subdirs.append(dirpath2)

    expanded_abnormal_path = os.path.expanduser(ABNORMAL_DATA_FOLDER_PATH)
    for dirname in os.listdir(expanded_abnormal_path):
        dirpath = os.path.join(expanded_abnormal_path, dirname)
        if not os.path.isdir(dirpath): continue
        print(f"Processing abnormal sequences from {dirpath}")
        subdirs.append(dirpath)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_and_store_sequences, subdir, file_parser)
            for subdir in subdirs
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing sequences: {e}")