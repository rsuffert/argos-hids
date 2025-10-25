"""
Module for processing the LID-DS dataset.

This module provides functionality to convert LID-DS syscall datasets into ML-ready HDF5 format.
It processes syscall traces from LID-DS scenarios and creates high-signal sequences for training.

For attack files, it extracts all syscalls from the attack timestamp onward. For normal files,
it processes the entire file into sequential 1024-syscall windows using a sliding window approach.

The processed data is split into training/validation/test sets and saved in HDF5 format compatible
with the DongTing dataset format for seamless integration with machine learning workflows.
"""

import os
import pickle
import zipfile
import json
import random
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add dataparse directory to path to import from dongting
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dongting.loader import append_seq_to_h5


def save_sequences_to_h5(sequences: List[List[int]], filepath: str, label_name: str = "") -> None:
    """
    Save multiple sequences to H5 file using DongTing's format.
    
    Creates an HDF5 file with variable-length sequences using gzip compression.
    This ensures compatibility with the DongTing dataset format and efficient storage.
    
    Args:
        sequences: List of syscall sequences (each sequence is a list of syscall IDs)
        filepath: Path where the H5 file will be saved
        label_name: Descriptive name for logging purposes (e.g., "attack training")
    """
    if not sequences:
        print(f"No sequences to save for {label_name}")
        return
    
    print(f"Saving {len(sequences)} {label_name} sequences to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Remove existing file to start fresh
    if os.path.exists(filepath):
        os.remove(filepath)
    
    for sequence in sequences:
        append_seq_to_h5(sequence, filepath)


class LIDDatasetLoader:
    """
    Main class for loading and processing LID dataset.
    
    This class handles the complete pipeline from raw LID-DS ZIP files to processed HDF5 files.
    It builds syscall dictionaries, processes files in parallel, extracts meaningful sequences,
    and saves the results in a format ready for machine learning training.
    
    The loader implements different strategies for attack vs normal data:
    - Attack data: Extracts sequences starting from the attack timestamp
    - Normal data: Uses sliding windows to capture comprehensive behavior patterns
    
    Attributes:
        WINDOW_SIZE: Maximum size for syscall sequences (1024)
        TRAIN_RATIO: Proportion of data for training (0.6)
        VAL_RATIO: Proportion of data for validation (0.2)
        MIN_CHUNK_SIZE: Minimum size for sequence chunks (50)
    """
    
    WINDOW_SIZE = 1024
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    MIN_CHUNK_SIZE = 50
    
    def __init__(self, data_dir: Optional[str] = None) -> None:
        """
        Initialize the LID dataset loader.
        
        Sets up paths and configuration for dataset processing. The data directory
        can be specified explicitly or through the LID_DATA_DIR environment variable.
        
        Args:
            data_dir: Directory containing LID dataset files. If None, uses LID_DATA_DIR 
                     environment variable or defaults to "lid-data".
        """
        self.data_dir: str = data_dir or os.getenv("LID_DATA_DIR") or "lid-data"
        self.syscall_dict_path = os.path.join(self.data_dir, "syscall_dict.pkl")
    
    def process_dataset(self) -> None:
        """
        Main method to process the entire dataset.
        
        Orchestrates the complete processing pipeline:
        1. Discovers ZIP files in the data directory
        2. Builds or loads syscall dictionary
        3. Processes all files in parallel
        4. Splits data into train/validation/test sets
        5. Saves results to HDF5 files
        
        The method provides progress updates and statistics throughout the process.
        """
        print("Starting LID-DS Loader")
        
        zip_files = [str(p) for p in Path(self.data_dir).rglob("*.zip") 
                    if "__MACOSX" not in str(p)]
        print(f"Found {len(zip_files)} ZIP files")
        
        if not zip_files:
            print("No ZIP files found. Please check your data directory.")
            return
        
        syscall_dict = self._build_syscall_dict(zip_files)
        sequences, labels = self._process_all_files(zip_files, syscall_dict)
        splits = self._split_data(sequences, labels)
        
        self._print_statistics(splits)
        self._save_to_h5(splits)
        print("Dataset processing completed")
    
    def _build_syscall_dict(self, zip_paths: List[str]) -> Dict[str, int]:
        """
        Build syscall dictionary from ZIP files.
        
        Creates a mapping from syscall names to integer IDs. This dictionary is cached
        to disk for reuse across runs. The method scans all ZIP files to collect
        unique syscall names and assigns sequential integer IDs.
        
        Args:
            zip_paths: List of paths to ZIP files containing syscall data
            
        Returns:
            Dictionary mapping syscall names to integer IDs
        """
        if os.path.exists(self.syscall_dict_path):
            with open(self.syscall_dict_path, "rb") as f_read:
                syscall_dict = pickle.load(f_read)
            print(f"Loaded existing dictionary with {len(syscall_dict)} entries")
            return syscall_dict
        
        print("Building new syscall dictionary...")
        all_syscalls = set()
        
        for i, zip_path in enumerate(zip_paths):
            if (i + 1) % 10 == 0:  # More frequent updates for small datasets
                print(f"Processed {i + 1}/{len(zip_paths)} files")
            
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    syscall_files = [f for f in zf.namelist() if f.endswith(".sc")]
                    if syscall_files:
                        with zf.open(syscall_files[0]) as sf:
                            for line in sf.readlines():
                                try:
                                    syscall_name = line.split()[5].decode()
                                    all_syscalls.add(syscall_name)
                                except (IndexError, UnicodeDecodeError):
                                    continue
            except Exception as e:
                print(f"Warning: Failed to process {zip_path}: {e}")
        
        syscall_dict = {name: i for i, name in enumerate(sorted(all_syscalls))}
        
        os.makedirs(os.path.dirname(self.syscall_dict_path), exist_ok=True)
        with open(self.syscall_dict_path, "wb") as f_write:
            pickle.dump(syscall_dict, f_write)
        
        print(f"Created dictionary with {len(syscall_dict)} syscalls")
        return syscall_dict
    
    def _process_zip_file(self, args: Tuple[str, Dict[str, int]]) -> Tuple[Optional[int], Optional[List[List[int]]]]:
        """
        Process a single ZIP file.
        
        Extracts metadata and syscall traces from a ZIP file, then creates sequences
        based on whether the file contains attack or normal data. For attack files,
        sequences start from the attack timestamp. For normal files, sliding windows
        are used to create multiple sequences.
        
        Args:
            args: Tuple containing (zip_path, syscall_dict)
            
        Returns:
            Tuple of (label, sequences) where label is 0/1 and sequences is a list
            of syscall ID sequences. Returns (None, None) if processing fails.
        """
        zip_path, syscall_dict = args
        
        try:
            metadata = self._extract_metadata(zip_path)
            if metadata is None:
                return None, None
            
            label = 1 if metadata.get("exploit") else 0
            attack_ts = self._get_attack_timestamp(metadata, label)
            
            parsed_syscalls = self._parse_syscalls(zip_path)
            if not parsed_syscalls:
                return None, None
            
            sequences = self._create_sequences(parsed_syscalls, label, attack_ts, syscall_dict)
            return label, sequences
            
        except Exception as e:
            print(f"Warning: Failed to process {zip_path}: {e}")
            return None, None

    def _extract_metadata(self, zip_path: str) -> Optional[Dict]:
        """Extract metadata from ZIP file."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            metadata_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not metadata_files:
                return None
            
            with zf.open(metadata_files[0]) as mf:
                return json.load(mf)

    def _get_attack_timestamp(self, metadata: Dict, label: int) -> Optional[int]:
        """Get attack timestamp from metadata."""
        if not label:
            return None
        
        exploit_info = metadata.get("time", {}).get("exploit", [])
        if not exploit_info or "absolute" not in exploit_info[0]:
            return None
        
        return int(str(exploit_info[0]["absolute"]).split(".")[0])

    def _parse_syscalls(self, zip_path: str) -> List[Tuple[int, str]]:
        """Parse syscalls from ZIP file."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            syscall_files = [f for f in zf.namelist() if f.endswith(".sc")]
            if not syscall_files:
                return []
            
            with zf.open(syscall_files[0]) as sf:
                lines = sf.readlines()
        
        parsed_syscalls = []
        for line in lines:
            try:
                timestamp = int(str(line.split()[0].decode()).split(".")[0])
                syscall_name = line.split()[5].decode()
                parsed_syscalls.append((timestamp, syscall_name))
            except (IndexError, UnicodeDecodeError):
                continue
        
        return parsed_syscalls

    def _create_sequences(self, parsed_syscalls: List[Tuple[int, str]], label: int, 
                         attack_ts: Optional[int], syscall_dict: Dict[str, int]) -> List[List[int]]:
        """Create sequences from parsed syscalls."""
        if label and attack_ts:
            return self._create_attack_sequences(parsed_syscalls, attack_ts, syscall_dict)
        return self._create_normal_sequences(parsed_syscalls, syscall_dict)

    def _create_attack_sequences(self, parsed_syscalls: List[Tuple[int, str]], 
                               attack_ts: int, syscall_dict: Dict[str, int]) -> List[List[int]]:
        """Create sequences for attack data starting from attack timestamp."""
        start_idx = 0
        for idx, (ts, _) in enumerate(parsed_syscalls):
            if ts >= attack_ts:
                start_idx = idx
                break
        
        syscall_names = [name for _, name in parsed_syscalls[start_idx:]]
        sequence = [syscall_dict[name] for name in syscall_names if name in syscall_dict]
        
        return [sequence[:self.WINDOW_SIZE]] if sequence else []

    def _create_normal_sequences(self, parsed_syscalls: List[Tuple[int, str]], 
                               syscall_dict: Dict[str, int]) -> List[List[int]]:
        """Create sequences for normal data using sliding windows."""
        sequences = []
        for i in range(0, len(parsed_syscalls), self.WINDOW_SIZE):
            chunk = parsed_syscalls[i:i + self.WINDOW_SIZE]
            if len(chunk) < self.MIN_CHUNK_SIZE and i > 0:
                continue
            
            syscall_names = [name for _, name in chunk]
            sequence = [syscall_dict[name] for name in syscall_names if name in syscall_dict]
            if sequence:
                sequences.append(sequence)
        
        return sequences
    
    def _process_all_files(
        self, zip_files: List[str], syscall_dict: Dict[str, int]
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Process all ZIP files in parallel.
        
        Uses multiprocessing to efficiently process large numbers of ZIP files.
        Collects all sequences and labels from individual file processing results.
        
        Args:
            zip_files: List of paths to ZIP files to process
            syscall_dict: Dictionary mapping syscall names to IDs
            
        Returns:
            Tuple of (all_sequences, all_labels) containing combined results
        """
        print("Processing files...")
        start_time = time.time()
        
        with Pool(cpu_count()) as pool:
            args = [(zip_path, syscall_dict) for zip_path in zip_files]
            results = pool.map(self._process_zip_file, args)
        
        attack_sequences = []
        normal_sequences = []
        
        for label, sequences_list in results:
            if label is not None and sequences_list:
                if label == 1:
                    attack_sequences.extend(sequences_list)
                else:
                    normal_sequences.extend(sequences_list)
        
        print(f"Found {len(attack_sequences)} attack, {len(normal_sequences)} normal sequences")
        print(f"Processing completed in {time.time() - start_time:.1f} seconds")
        
        all_sequences = attack_sequences + normal_sequences
        all_labels = [1] * len(attack_sequences) + [0] * len(normal_sequences)
        
        return all_sequences, all_labels
    
    def _split_data(
        self, sequences: List[List[int]], labels: List[int]
    ) -> Dict[str, Tuple[List[List[int]], List[int]]]:
        """
        Split data into train/val/test sets.
        
        Randomly shuffles the data and splits it according to predefined ratios.
        Uses a fixed random seed for reproducible splits across runs.
        
        Args:
            sequences: List of all syscall sequences
            labels: List of corresponding labels (0 for normal, 1 for attack)
            
        Returns:
            Dictionary with keys 'training', 'validation', 'test' and values
            containing tuples of (sequences, labels) for each split
        """
        data = list(zip(sequences, labels, strict=True))
        random.seed(42)
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * self.TRAIN_RATIO)
        val_end = train_end + int(total * self.VAL_RATIO)
        
        splits = {}
        for name, split_data in [
            ("training", data[:train_end]),
            ("validation", data[train_end:val_end]),
            ("test", data[val_end:])
        ]:
            if split_data:
                seqs, lbls = zip(*split_data, strict=True)
                splits[name] = (list(seqs), list(lbls))
            else:
                splits[name] = ([], [])
        
        return splits
    
    def _print_statistics(self, splits: Dict[str, Tuple[List[List[int]], List[int]]]) -> None:
        """
        Print statistics for data splits.
        
        Displays the number of attack and normal sequences in each split
        for verification and debugging purposes.
        
        Args:
            splits: Dictionary containing train/validation/test splits
        """
        print("Data splits:")
        for split_name in ["training", "validation", "test"]:
            seqs, lbls = splits[split_name]
            attack_count = sum(lbls)
            normal_count = len(lbls) - attack_count
            print(f"  {split_name}: {attack_count} attacks, {normal_count} normal")
    
    def _save_to_h5(self, splits: Dict[str, Tuple[List[List[int]], List[int]]]) -> None:
        """
        Save data splits to H5 files using DongTing's format.
        
        Creates separate HDF5 files for each label-split combination. Files are
        named using the convention: {label}_{split}.h5 where label is 0 for normal
        and 1 for attack data.
        
        Args:
            splits: Dictionary containing train/validation/test splits with sequences and labels
        """
        for split_name in ["training", "validation", "test"]:
            seqs, lbls = splits[split_name]
            
            # Separate by label
            attack_seqs = [s for s, label in zip(seqs, lbls, strict=True) if label == 1]
            normal_seqs = [s for s, label in zip(seqs, lbls, strict=True) if label == 0]
            
            # Save using DongTing's format
            if attack_seqs:
                filepath = os.path.join(self.data_dir, f"1_{split_name}.h5")
                save_sequences_to_h5(attack_seqs, filepath, f"attack {split_name}")
            
            if normal_seqs:
                filepath = os.path.join(self.data_dir, f"0_{split_name}.h5")
                save_sequences_to_h5(normal_seqs, filepath, f"normal {split_name}")


def main() -> None:
    """Entry point for the LID dataset loader."""
    loader = LIDDatasetLoader()
    loader.process_dataset()


if __name__ == "__main__":
    main()
