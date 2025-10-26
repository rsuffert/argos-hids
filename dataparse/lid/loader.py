"""
LID-DS dataset loader for converting syscall traces to ML-ready HDF5 format.
Processes attack and normal data with different strategies for optimal training.
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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dongting.loader import append_seq_to_h5


def save_sequences_to_h5(sequences: List[List[int]], filepath: str, label_name: str = "") -> None:
    """Save sequences to H5 file using DongTing's format."""
    if not sequences:
        print(f"No sequences to save for {label_name}")
        return
    
    print(f"Saving {len(sequences)} {label_name} sequences to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    for sequence in sequences:
        append_seq_to_h5(sequence, filepath)


class LIDDatasetLoader:
    """Load and process LID dataset to HDF5 format."""
    
    WINDOW_SIZE = 1024
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    MIN_CHUNK_SIZE = 50
    
    def __init__(self, data_dir: Optional[str] = None) -> None:
        """Initialize loader with data directory."""
        self.data_dir = data_dir or os.getenv("LID_DATA_DIR") or "lid-data"
        self.syscall_dict_path = os.path.join(self.data_dir, "syscall_dict.pkl")
    
    def process_dataset(self) -> None:
        """Main processing pipeline."""
        print("Starting LID-DS Loader")
        
        zip_files = self._find_zip_files()
        if not zip_files:
            print("No ZIP files found. Please check your data directory.")
            return
        
        syscall_dict = self._build_syscall_dict(zip_files)
        sequences, labels = self._process_all_files(zip_files, syscall_dict)
        splits = self._split_data(sequences, labels)
        
        self._print_statistics(splits)
        self._save_to_h5(splits)
        print("Dataset processing completed")
    
    def _find_zip_files(self) -> List[str]:
        """Find ZIP files in data directory."""
        zip_files = [str(p) for p in Path(self.data_dir).rglob("*.zip") 
                    if "__MACOSX" not in str(p)]
        print(f"Found {len(zip_files)} ZIP files")
        return zip_files
    
    def _build_syscall_dict(self, zip_paths: List[str]) -> Dict[str, int]:
        """Build or load syscall dictionary."""
        if os.path.exists(self.syscall_dict_path):
            return self._load_cached_dict()
        return self._create_new_dict(zip_paths)

    def _load_cached_dict(self) -> Dict[str, int]:
        """Load cached syscall dictionary."""
        with open(self.syscall_dict_path, "rb") as f:
            syscall_dict = pickle.load(f)
        print(f"Loaded existing dictionary with {len(syscall_dict)} entries")
        return syscall_dict

    def _create_new_dict(self, zip_paths: List[str]) -> Dict[str, int]:
        """Create new syscall dictionary."""
        print("Building new syscall dictionary...")
        all_syscalls = set()
        
        for i, zip_path in enumerate(zip_paths):
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(zip_paths)} files")
            
            syscalls = self._extract_syscalls_from_zip(zip_path)
            all_syscalls.update(syscalls)
        
        syscall_dict = {name: i for i, name in enumerate(sorted(all_syscalls))}
        self._save_dict_to_cache(syscall_dict)
        
        print(f"Created dictionary with {len(syscall_dict)} syscalls")
        return syscall_dict

    def _extract_syscalls_from_zip(self, zip_path: str) -> set:
        """Extract syscalls from ZIP file."""
        syscalls: set[str] = set()
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                syscall_files = [f for f in zf.namelist() if f.endswith(".sc")]
                if not syscall_files:
                    return syscalls
                
                with zf.open(syscall_files[0]) as sf:
                    for line in sf.readlines():
                        syscall_name = self._parse_syscall_name(line)
                        if syscall_name:
                            syscalls.add(syscall_name)
        except Exception as e:
            print(f"Warning: Failed to process {zip_path}: {e}")
        
        return syscalls

    def _parse_syscall_name(self, line: bytes) -> Optional[str]:
        """Parse syscall name from line."""
        try:
            return line.split()[5].decode()
        except (IndexError, UnicodeDecodeError):
            return None

    def _save_dict_to_cache(self, syscall_dict: Dict[str, int]) -> None:
        """Save dictionary to cache."""
        os.makedirs(os.path.dirname(self.syscall_dict_path), exist_ok=True)
        with open(self.syscall_dict_path, "wb") as f:
            pickle.dump(syscall_dict, f)
    
    def _process_zip_file(self, args: Tuple[str, Dict[str, int]]) -> Tuple[Optional[int], Optional[List[List[int]]]]:
        """Process single ZIP file."""
        zip_path, syscall_dict = args
        
        try:
            metadata = self._extract_metadata(zip_path)
            if not metadata:
                return None, None
            
            return self._process_valid_zip(zip_path, metadata, syscall_dict)
        except Exception as e:
            print(f"Warning: Failed to process {zip_path}: {e}")
            return None, None

    def _process_valid_zip(self, zip_path: str, metadata: Dict, syscall_dict: Dict[str, int]) -> Tuple[int, List[List[int]]]:
        """Process validated ZIP file."""
        label = 1 if metadata.get("exploit") else 0
        attack_ts = self._get_attack_timestamp(metadata, label)
        
        parsed_syscalls = self._parse_syscalls(zip_path)
        if not parsed_syscalls:
            return label, []
        
        sequences = self._create_sequences(parsed_syscalls, label, attack_ts, syscall_dict)
        return label, sequences

    def _extract_metadata(self, zip_path: str) -> Optional[Dict]:
        """Extract metadata from ZIP."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            metadata_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not metadata_files:
                return None
            
            with zf.open(metadata_files[0]) as mf:
                return json.load(mf)

    def _get_attack_timestamp(self, metadata: Dict, label: int) -> Optional[int]:
        """Get attack timestamp."""
        if not label:
            return None
        
        exploit_info = metadata.get("time", {}).get("exploit", [])
        if not exploit_info or "absolute" not in exploit_info[0]:
            return None
        
        return int(str(exploit_info[0]["absolute"]).split(".")[0])

    def _parse_syscalls(self, zip_path: str) -> List[Tuple[int, str]]:
        """Parse syscalls from ZIP."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            syscall_files = [f for f in zf.namelist() if f.endswith(".sc")]
            if not syscall_files:
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

    def _create_sequences(self, parsed_syscalls: List[Tuple[int, str]], label: int, 
                         attack_ts: Optional[int], syscall_dict: Dict[str, int]) -> List[List[int]]:
        """Create sequences from syscalls."""
        if label and attack_ts:
            return self._create_attack_sequences(parsed_syscalls, attack_ts, syscall_dict)
        return self._create_normal_sequences(parsed_syscalls, syscall_dict)

    def _create_attack_sequences(self, parsed_syscalls: List[Tuple[int, str]], 
                               attack_ts: int, syscall_dict: Dict[str, int]) -> List[List[int]]:
        """Create attack sequences."""
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
        """Create normal sequences."""
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
    
    def _process_all_files(self, zip_files: List[str], syscall_dict: Dict[str, int]) -> Tuple[List[List[int]], List[int]]:
        """Process all files in parallel."""
        print("Processing files...")
        start_time = time.time()
        
        with Pool(cpu_count()) as pool:
            args = [(zip_path, syscall_dict) for zip_path in zip_files]
            results = pool.map(self._process_zip_file, args)
        
        attack_sequences: List[List[int]] = []
        normal_sequences: List[List[int]] = []
        
        for label, sequences_list in results:
            if label is not None and sequences_list:
                target = attack_sequences if label == 1 else normal_sequences
                target.extend(sequences_list)
        
        all_sequences = attack_sequences + normal_sequences
        all_labels = [1] * len(attack_sequences) + [0] * len(normal_sequences)
        
        elapsed = time.time() - start_time
        print(f"Found {len(attack_sequences)} attack, {len(normal_sequences)} normal sequences")
        print(f"Processing completed in {elapsed:.1f} seconds")
        
        return all_sequences, all_labels

    def _split_data(self, sequences: List[List[int]], labels: List[int]) -> Dict[str, Tuple[List[List[int]], List[int]]]:
        """Split data into train/val/test."""
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
        """Print split statistics."""
        print("Data splits:")
        for split_name in ["training", "validation", "test"]:
            seqs, lbls = splits[split_name]
            attack_count = sum(lbls)
            normal_count = len(lbls) - attack_count
            print(f"  {split_name}: {attack_count} attacks, {normal_count} normal")
    
    def _save_to_h5(self, splits: Dict[str, Tuple[List[List[int]], List[int]]]) -> None:
        """Save splits to H5 files."""
        for split_name in ["training", "validation", "test"]:
            seqs, lbls = splits[split_name]
            
            attack_seqs = [s for s, l in zip(seqs, lbls, strict=True) if l == 1]
            normal_seqs = [s for s, l in zip(seqs, lbls, strict=True) if l == 0]
            
            if attack_seqs:
                filepath = os.path.join(self.data_dir, f"1_{split_name}.h5")
                save_sequences_to_h5(attack_seqs, filepath, f"attack {split_name}")
            
            if normal_seqs:
                filepath = os.path.join(self.data_dir, f"0_{split_name}.h5")
                save_sequences_to_h5(normal_seqs, filepath, f"normal {split_name}")


def main() -> None:
    """Entry point."""
    loader = LIDDatasetLoader()
    loader.process_dataset()


if __name__ == "__main__":
    main()
