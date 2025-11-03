"""
Module for parsing system call sequences from LID-DS dataset raw files and storing them
in HDF5 compressed and pre-processed format for seamless integration with machine learning
workflows.
"""

import os
import pickle
import zipfile
import json
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys
import csv
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dongting.loader import append_seq_to_h5

SEQUENCE_LENGTH = 1024
MIN_CHUNK_LENGTH = 50
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 42

SYSCALL_MAPPING_DUMP_PATH = "mapping.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def save_sequences_to_h5(sequences: List[List[int]], filepath: str) -> None:
    """
    Save multiple sequences to an HDF5 file using DongTing's format.

    Args:
        sequences (List[List[int]]): List of syscall sequences to save.
        filepath (str): Path where the H5 file will be saved.
    """
    if not sequences:
        return
    
    logger.info(f"Saving {len(sequences)} sequences to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    for sequence in sequences:
        append_seq_to_h5(sequence, filepath)


class LIDDatasetLoader:
    """
    Load and process LID dataset into HDF5 format for machine learning workflows.
    
    This class handles the complete pipeline from raw LID-DS ZIP files to processed HDF5 files.
    It builds syscall dictionaries, processes files in parallel, extracts meaningful sequences,
    and saves the results in a format ready for machine learning training.
    """
    
    def __init__(self, data_dir: Optional[str] = None) -> None:
        """
        Initialize LID dataset loader.
        
        Args:
            data_dir (Optional[str]): Path to the data directory. If None, uses LID_DATA_DIR 
                environment variable or defaults to 'lid-data'.
        """
        self.data_dir: str = data_dir or os.getenv("LID_DATA_DIR") or "lid-data"
        self.syscall_dict_path = os.path.join(self.data_dir, "syscall_dict.pkl")
    
    def process_dataset(self) -> None:
        """
        Main processing pipeline for the entire dataset.
        
        Orchestrates the complete processing pipeline:
        1. Discovers ZIP files in the data directory
        2. Builds or loads syscall dictionary
        3. Dumps a CSV mapping of syscall names to IDs
        4. Processes all files in parallel
        5. Splits data into train/validation/test sets
        6. Saves results to HDF5 files
        """
        logger.info("Starting LID-DS Loader")
        
        if not self.data_dir:
            logger.info("No data directory specified")
            return
        
        zip_files = [str(p) for p in Path(self.data_dir).rglob("*.zip") 
                    if "__MACOSX" not in str(p)]
        
        if not zip_files:
            logger.info("No ZIP files found")
            return
        
        logger.info(f"Found {len(zip_files)} ZIP files")

        syscall_dict = self._get_syscall_dict(zip_files)
        
        # dump syscall mapping to csv
        csv_path = os.path.join(self.data_dir, SYSCALL_MAPPING_DUMP_PATH)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(syscall_dict.items())
        logger.info(f"Dumped syscall mapping to {csv_path}")
        
        sequences, labels = self._process_files(zip_files, syscall_dict)
        splits = self._split_data(sequences, labels)
        self._save_splits(splits)
        
        logger.info("Processing completed")
    
    def _get_syscall_dict(self, zip_files: List[str]) -> Dict[str, int]:
        """
        Build or load syscall dictionary from ZIP files.
        
        Creates a mapping from syscall names to integer IDs. This dictionary is cached
        to disk for reuse across runs.
        
        Args:
            zip_files (List[str]): List of paths to ZIP files containing syscall data.
            
        Returns:
            Dict[str, int]: Dictionary mapping syscall names to integer IDs.
        """
        if os.path.exists(self.syscall_dict_path):
            with open(self.syscall_dict_path, "rb") as f:
                syscall_dict = pickle.load(f)
            logger.info(f"Loaded existing dictionary with {len(syscall_dict)} entries")
            return syscall_dict
        
        logger.info("Building syscall dictionary...")
        all_syscalls = set()
        
        for zip_path in zip_files:
            syscalls = self._extract_syscalls_from_zip(zip_path)
            all_syscalls.update(syscalls)
        
        syscall_dict = {name: i for i, name in enumerate(sorted(all_syscalls))}
        os.makedirs(os.path.dirname(self.syscall_dict_path), exist_ok=True)
        with open(self.syscall_dict_path, "wb") as f:
            pickle.dump(syscall_dict, f)
        logger.info(f"Created dictionary with {len(syscall_dict)} syscalls")
        return syscall_dict
    
    def _extract_syscalls_from_zip(self, zip_path: str) -> set:
        """Extract syscalls from a single ZIP file."""
        syscalls: set[str] = set()
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                sc_files = [fname for fname in zf.namelist() if fname.endswith(".sc")]
                if not sc_files:
                    return syscalls
                
                with zf.open(sc_files[0]) as scf:
                    for line in scf.readlines():
                        try:
                            syscall = line.split()[5].decode()
                            syscalls.add(syscall)
                        except (IndexError, UnicodeDecodeError):
                            pass
        except Exception:
            pass
        return syscalls
    
    def _process_single_file(self, args: Tuple[str, Dict[str, int]]) -> Tuple[Optional[int], List[List[int]]]:
        """
        Process a single ZIP file and extract sequences.
        
        Args:
            args (Tuple[str, Dict[str, int]]): Tuple containing zip_path and syscall_dict.
            
        Returns:
            Tuple[Optional[int], List[List[int]]]: Tuple of (label, sequences) where label 
                is 0/1 and sequences is a list of syscall ID sequences. Returns (None, []) 
                if processing fails.
        """
        zip_path, syscall_dict = args
        
        try:
            metadata, lines = self._extract_zip_data(zip_path)
            if metadata is None:
                return None, []
            
            label = 1 if metadata.get("exploit") else 0
            attack_ts = self._get_attack_timestamp(metadata, label)
            parsed = self._parse_syscall_lines(lines)
            sequences = self._create_sequences(parsed, label, attack_ts, syscall_dict)
            
            return label, sequences
            
        except Exception:
            # check if file is missing/corrupted
            return None, []
    
    def _extract_zip_data(self, zip_path: str) -> Tuple[Optional[Dict], List]:
        """
        Extract metadata and syscall lines from ZIP file.
        
        Args:
            zip_path (str): Path to the ZIP file.
            
        Returns:
            Tuple[Optional[Dict], List]: Tuple of (metadata_dict, syscall_lines). 
                Returns (None, []) if extraction fails.
        """
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Get metadata
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not json_files:
                return None, []
            
            with zf.open(json_files[0]) as mf:
                metadata = json.load(mf)
            
            # Get syscalls
            sc_files = [f for f in zf.namelist() if f.endswith(".sc")]
            if not sc_files:
                return None, []
            
            with zf.open(sc_files[0]) as sf:
                lines = sf.readlines()
        
        return metadata, lines
    
    def _get_attack_timestamp(self, metadata: Dict, label: int) -> Optional[int]:
        """
        Extract attack timestamp from metadata.
        
        Args:
            metadata (Dict): Metadata dictionary from JSON file.
            label (int): Label indicating if this is an attack (1) or normal (0).
            
        Returns:
            Optional[int]: Attack timestamp if available, None otherwise.
        """
        if not label:
            return None
        
        exploit_info = metadata.get("time", {}).get("exploit", [])
        if not exploit_info or "absolute" not in exploit_info[0]:
            return None
        
        return int(str(exploit_info[0]["absolute"]).split(".")[0])
    
    def _parse_syscall_lines(self, lines: List) -> List[Tuple[int, str]]:
        """
        Parse syscall lines into timestamp-name pairs.
        
        Args:
            lines (List): List of raw syscall lines from .sc file.
            
        Returns:
            List[Tuple[int, str]]: List of (timestamp, syscall_name) tuples.
        """
        parsed = []
        for line in lines:
            try:
                parts = line.split()
                ts = int(str(parts[0].decode()).split(".")[0])
                name = parts[5].decode()
                parsed.append((ts, name))
            except (IndexError, UnicodeDecodeError):
                pass
        return parsed
    
    def _create_sequences(self, parsed: List[Tuple[int, str]], label: int, 
                         attack_ts: Optional[int], syscall_dict: Dict[str, int]) -> List[List[int]]:
        """
        Create sequences from parsed syscall data.
        
        Args:
            parsed (List[Tuple[int, str]]): List of (timestamp, syscall_name) tuples.
            label (int): Label indicating attack (1) or normal (0).
            attack_ts (Optional[int]): Attack timestamp if available.
            syscall_dict (Dict[str, int]): Dictionary mapping syscall names to IDs.
            
        Returns:
            List[List[int]]: List of syscall ID sequences.
        """
        return (self._create_attack_sequences(parsed, attack_ts, syscall_dict) 
                if label and attack_ts 
                else self._create_normal_sequences(parsed, syscall_dict))
    
    def _create_attack_sequences(self, parsed: List[Tuple[int, str]], 
                               attack_ts: int, syscall_dict: Dict[str, int]) -> List[List[int]]:
        """
        Create attack sequences starting from attack timestamp.
        
        Args:
            parsed (List[Tuple[int, str]]): List of (timestamp, syscall_name) tuples.
            attack_ts (int): Attack timestamp to start from.
            syscall_dict (Dict[str, int]): Dictionary mapping syscall names to IDs.
            
        Returns:
            List[List[int]]: List containing single attack sequence.
        """
        start_idx = next((i for i, (ts, _) in enumerate(parsed) if ts >= attack_ts), 0)
        names = [name for _, name in parsed[start_idx:]]
        seq = [syscall_dict[name] for name in names if name in syscall_dict]
        return [seq[:SEQUENCE_LENGTH]] if seq else []
    
    def _create_normal_sequences(self, parsed: List[Tuple[int, str]], 
                               syscall_dict: Dict[str, int]) -> List[List[int]]:
        """
        Create normal sequences using sliding window approach.
        
        Args:
            parsed (List[Tuple[int, str]]): List of (timestamp, syscall_name) tuples.
            syscall_dict (Dict[str, int]): Dictionary mapping syscall names to IDs.
            
        Returns:
            List[List[int]]: List of normal sequences.
        """
        sequences = []
        for i in range(0, len(parsed), SEQUENCE_LENGTH):
            chunk = parsed[i:i + SEQUENCE_LENGTH]
            if len(chunk) >= MIN_CHUNK_LENGTH or i == 0:
                names = [name for _, name in chunk]
                seq = [syscall_dict[name] for name in names if name in syscall_dict]
                if seq:
                    sequences.append(seq)
        return sequences
    
    def _process_files(self, zip_files: List[str], syscall_dict: Dict[str, int]) -> Tuple[List[List[int]], List[int]]:
        """
        Process all ZIP files in parallel.
        
        Args:
            zip_files (List[str]): List of paths to ZIP files to process.
            syscall_dict (Dict[str, int]): Dictionary mapping syscall names to IDs.
            
        Returns:
            Tuple[List[List[int]], List[int]]: Tuple of (all_sequences, all_labels).
        """
        logger.info("Processing files...")
        
        with Pool(cpu_count()) as pool:
            args = [(f, syscall_dict) for f in zip_files]
            results = pool.map(self._process_single_file, args)
        
        all_sequences = []
        all_labels = []
        
        for label, sequences in results:
            if label is not None:
                all_sequences.extend(sequences)
                all_labels.extend([label] * len(sequences))
        
        attack_count = sum(all_labels)
        normal_count = len(all_labels) - attack_count
        logger.info(f"Found {attack_count} attack, {normal_count} normal sequences")
        
        return all_sequences, all_labels
    
    def _split_data(
        self, sequences: List[List[int]], labels: List[int]
    ) -> Dict[str, Tuple[List[List[int]], List[int]]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            sequences (List[List[int]]): List of all syscall sequences.
            labels (List[int]): List of corresponding labels (0 for normal, 1 for attack).
            
        Returns:
            Dict[str, Tuple[List[List[int]], List[int]]]: Dictionary with keys 'training', 
                'validation', 'test' and values containing tuples of (sequences, labels).
        """
        data = list(zip(sequences, labels, strict=True))
        random.seed(RANDOM_SEED)
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)
        
        splits = {
            "training": data[:train_end],
            "validation": data[train_end:val_end],
            "test": data[val_end:]
        }
        
        result = {}
        for name, split_data in splits.items():
            if split_data:
                seqs, lbls = zip(*split_data, strict=True)
                result[name] = (list(seqs), list(lbls))
            else:
                result[name] = ([], [])
        
        return result
    
    def _save_splits(self, splits: Dict[str, Tuple[List[List[int]], List[int]]]) -> None:
        """
        Save data splits to HDF5 files.
        
        Args:
            splits (Dict[str, Tuple[List[List[int]], List[int]]]): Dictionary containing 
                train/validation/test splits with sequences and labels.
        """
        for split_name, (seqs, lbls) in splits.items():
            attack_seqs = [s for s, label in zip(seqs, lbls, strict=True) if label == 1]
            normal_seqs = [s for s, label in zip(seqs, lbls, strict=True) if label == 0]

            if attack_seqs:
                path = os.path.join(self.data_dir, f"1_{split_name}.h5")
                logger.info(f"Saving {len(attack_seqs)} attack {split_name} sequences to {path}")
                save_sequences_to_h5(attack_seqs, path)

            if normal_seqs:
                path = os.path.join(self.data_dir, f"0_{split_name}.h5")
                logger.info(f"Saving {len(normal_seqs)} normal {split_name} sequences to {path}")
                save_sequences_to_h5(normal_seqs, path)


def main() -> None:
    """Entry point for the LID dataset loader."""
    loader = LIDDatasetLoader()
    loader.process_dataset()


if __name__ == "__main__":
    main()
