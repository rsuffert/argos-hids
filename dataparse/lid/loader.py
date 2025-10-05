"""
LID Dataset Loader for ARGOS-HIDS
Supports H5 and LID-DS formats for GNN training.
"""

import os
import json
import logging
import argparse
import zipfile
import tempfile
import h5py
from pathlib import Path
from typing import List, Optional, Tuple

# GNN module import from preprocessing
try:
    from preprocessing.graph_preprocess_dataset import preprocess_dataset # type: ignore
    HAS_PREPROCESSING = True
except ImportError:
    HAS_PREPROCESSING = False
    logging.warning("preprocessing.graph_preprocess_dataset not available, preprocessing disabled")

PKL_TRACES_FILENAME = "processed_graphs.pkl"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Syscall Classification Pipeline")
    parser.add_argument("-l", "--lidds", type=str, help="Path to LID-DS dataset directory")
    parser.add_argument("-n", "--normal", type=str, help="Path to normal traces H5 file")
    parser.add_argument("-a", "--attack", type=str, help="Path to attack traces H5 file")
    parser.add_argument("-p", "--preprocess", action="store_true", help="Preprocess training data to graphs")
    return parser.parse_args()


def create_syscall_windows(syscalls: List[str], is_attack: bool = False) -> List[List[str]]:
    """Create windowed sequences with adaptive sizing."""
    if len(syscalls) < 20:
        return []
    
    # Simple window logic
    if is_attack:
        max_windows = 10
        window_size = min(1024, len(syscalls))
    else:
        max_windows = 50
        window_size = min(1024, len(syscalls))
    
    stride = max(1, window_size // 2)
    windows = [syscalls[i:i+window_size] for i in range(0, len(syscalls)-window_size+1, stride)]
    
    return windows[:max_windows] if windows else [syscalls]


def save_traces_to_files(traces: List[List[str]], output_directory: str, is_attack: bool = False) -> None:
    """Save syscall traces to text files with windowing."""
    os.makedirs(output_directory, exist_ok=True)
    file_counter = 0
    
    for trace in traces:
        for window in create_syscall_windows(trace, is_attack):
            with open(f"{output_directory}/trace_{file_counter}.txt", "w") as f:
                f.write("\n".join(f"{syscall}(" for syscall in window))
            file_counter += 1
    
    label = "Attack" if is_attack else "Normal"
    logging.info(f"{label}: {len(traces)} traces → {file_counter} files")


def extract_h5_traces(h5_path: str) -> List[List[str]]:
    """Extract syscall traces from H5 file."""
    try:
        with h5py.File(h5_path, "r") as file:
            data = file["sequences"]
            traces = [[str(int(s)) for s in data[i] if s != 0] for i in range(data.shape[0])]
            return [t for t in traces if len(t) >= 20]
    except Exception as error:
        logging.error(f"Error processing H5 file {h5_path}: {error}")
        return []


def split_dataset(traces: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    """Split dataset into train and inference sets (70/30)."""
    if not traces:
        return [], []
    split_idx = int(0.7 * len(traces))
    return traces[:split_idx], traces[split_idx:]


def process_h5_dataset(normal_h5_path: str, attack_h5_path: str) -> None:
    """Process H5 format datasets."""
    logging.info(f"Processing H5 - Normal: {normal_h5_path}, Attack: {attack_h5_path}")
    
    normal_traces = extract_h5_traces(normal_h5_path) if normal_h5_path and os.path.exists(normal_h5_path) else []
    attack_traces = extract_h5_traces(attack_h5_path) if attack_h5_path and os.path.exists(attack_h5_path) else []
    
    train_normal, infer_normal = split_dataset(normal_traces)
    train_attack, infer_attack = split_dataset(attack_traces)
    
    logging.info(f"Split - Train: {len(train_normal)} normal / {len(train_attack)} attack")
    
    # Save traces
    for traces, path, is_attack in [
        (train_normal, "traces_train/normal", False),
        (train_attack, "traces_train/attack", True),
        (infer_normal, "traces_infer/normal", False),
        (infer_attack, "traces_infer/attack", True)
    ]:
        save_traces_to_files(traces, path, is_attack)


def should_skip_file(path_str: str) -> bool:
    """Check if file should be skipped."""
    return "__MACOSX" in path_str or ".DS_Store" in path_str


def extract_zip_files(dataset_path: str, temp_dir: str) -> Tuple[List[Path], dict]:
    """Extract ZIP files and return syscall files and metadata."""
    syscall_files, json_metadata = [], {}
    
    for zip_path in Path(dataset_path).rglob("*.zip"):
        if should_skip_file(str(zip_path)):
            continue
            
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                for file_info in zip_file.filelist:
                    if should_skip_file(file_info.filename) or not file_info.filename.endswith((".sc", ".json")):
                        continue
                        
                    stem = Path(file_info.filename).stem
                    ext = Path(file_info.filename).suffix
                    extracted_path = Path(temp_dir) / f"{stem}{ext}"
                    extracted_path.write_bytes(zip_file.read(file_info))
                    
                    if ext == ".sc":
                        syscall_files.append(extracted_path)
                    else:  # .json
                        try:
                            with open(extracted_path, "r") as f:
                                json_metadata[stem] = json.load(f)
                        except Exception:
                            pass
        except Exception as e:
            logging.warning(f"Error processing {zip_path}: {e}")
    
    return syscall_files, json_metadata


def parse_syscall_file(syscall_file: Path) -> List[Tuple[int, str]]:
    """Parse syscall file and return list of (timestamp, syscall) tuples."""
    try:
        with open(syscall_file, "r") as f:
            syscalls = []
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                
                # Try timestamp parsing, fallback to line number
                try:
                    timestamp = int(parts[0]) if len(parts) >= 2 else line_num
                    syscall = parts[1] if len(parts) >= 2 else parts[0]
                except ValueError:
                    timestamp, syscall = line_num, parts[0]
                
                syscalls.append((timestamp, syscall))
            return syscalls
    except Exception as e:
        logging.warning(f"Error parsing syscall file {syscall_file}: {e}")
        return []


def extract_traces_from_syscalls(
    all_syscalls: List[Tuple[int, str]], 
    is_exploit: bool, 
    attack_timestamp: Optional[int]
) -> Tuple[List[str], List[str]]:
    """Extract normal and attack traces from syscall data."""
    syscalls = [sc for _, sc in all_syscalls]
    
    if len(syscalls) < 20:
        return [], []
    
    if not is_exploit:
        return syscalls, []
    
    if not attack_timestamp:
        return [], syscalls
    
    # Find attack start
    attack_start = _find_attack_start(all_syscalls, attack_timestamp)
    return _split_traces_at_attack(all_syscalls, attack_start)


def _find_attack_start(all_syscalls: List[Tuple[int, str]], attack_timestamp: int) -> int:
    """Find index where attack starts."""
    for i, (ts, _) in enumerate(all_syscalls):
        if ts >= attack_timestamp:
            return i
    return len(all_syscalls)


def _split_traces_at_attack(all_syscalls: List[Tuple[int, str]], attack_start: int) -> Tuple[List[str], List[str]]:
    """Split syscalls into normal and attack traces."""
    normal_trace = [sc for _, sc in all_syscalls[:attack_start]] if attack_start > 20 else []
    attack_trace = [sc for _, sc in all_syscalls[attack_start:]]
    
    return (
        normal_trace if len(normal_trace) >= 20 else [],
        attack_trace if len(attack_trace) >= 20 else []
    )


def process_single_syscall_file(
    syscall_file: Path, json_metadata: dict
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Process a single syscall file and return traces."""
    stem = syscall_file.stem
    metadata = json_metadata.get(stem, {})
    is_exploit = metadata.get("exploit", False)
    attack_timestamp = metadata.get("attack_timestamp") if is_exploit else None
    
    all_syscalls = parse_syscall_file(syscall_file)
    if not all_syscalls:
        return None, None
    
    return extract_traces_from_syscalls(all_syscalls, is_exploit, attack_timestamp)


def process_lidds_dataset(dataset_path: str) -> None:
    """Process LID-DS format dataset with timestamp-based attack extraction."""
    logging.info(f"Processing LID-DS dataset: {dataset_path}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        syscall_files, json_metadata = extract_zip_files(dataset_path, temp_dir)
        
        exploit_count = sum(1 for meta in json_metadata.values() if meta.get("exploit", False))
        logging.info(f"Extracted {len(syscall_files)} .sc files, {exploit_count} exploits")
        
        normal_traces, attack_traces = [], []
        
        for syscall_file in syscall_files:
            normal_trace, attack_trace = process_single_syscall_file(syscall_file, json_metadata)
            
            if normal_trace:
                normal_traces.append(normal_trace)
            if attack_trace:
                attack_traces.append(attack_trace)
        
        logging.info(f"Extracted {len(attack_traces)} attacks, {len(normal_traces)} normal traces")
        
        if not normal_traces and not attack_traces:
            logging.error("No valid traces found!")
            return
        
        train_normal, infer_normal = split_dataset(normal_traces)
        train_attack, infer_attack = split_dataset(attack_traces)
        
        logging.info(f"Train: {len(train_normal)} normal / {len(train_attack)} attack")
        logging.info(f"Infer: {len(infer_normal)} normal / {len(infer_attack)} attack")
        
        # Save traces
        for traces, path, is_attack in [
            (train_normal, "traces_train/normal", False),
            (train_attack, "traces_train/attack", True),
            (infer_normal, "traces_infer/normal", False),
            (infer_attack, "traces_infer/attack", True)
        ]:
            save_traces_to_files(traces, path, is_attack)


def main() -> None:
    """Main execution function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    args = parse_arguments()
    
    if args.lidds:
        process_lidds_dataset(args.lidds)
        
        if args.preprocess:
            output_path = preprocess_dataset("traces_train", False, False, PKL_TRACES_FILENAME)
            logging.info(f"Graph preprocessing completed: {output_path}")
            
    elif args.normal or args.attack:
        normal_h5 = args.normal or os.getenv("LID_NORMAL")
        attack_h5 = args.attack or os.getenv("LID_ATTACK")
        
        if normal_h5 and attack_h5:
            process_h5_dataset(normal_h5, attack_h5)
            
            if args.preprocess:
                output_path = preprocess_dataset("traces_train", False, False, PKL_TRACES_FILENAME)
                logging.info(f"Graph preprocessing completed: {output_path}")
        else:
            logging.error("Both normal and attack H5 files must be specified")
    else:
        logging.error("No dataset specified. Use -l for LID-DS or -n/-a for H5 files")


if __name__ == "__main__":
    main()