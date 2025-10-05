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
from typing import List

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


def _get_attack_window_config(syscall_length: int) -> tuple[int, int] | None:
    """Get window size and stride for attack traces based on length."""
    if syscall_length >= 1024:
        return 1024, 512
    if syscall_length >= 512:
        return 512, 256
    return None


def _create_attack_windows(syscalls: List[str]) -> List[List[str]]:
    """Create windows for attack traces with adaptive sizing."""
    length = len(syscalls)
    config = _get_attack_window_config(length)
    
    if config is None:
        if length >= 256:
            return [syscalls[:256]]
        return [syscalls]
    
    window_size, stride = config
    windows = []
    for start in range(0, length - window_size + 1, stride):
        windows.append(syscalls[start:start + window_size])
    return windows[:10]  # Max 10 windows per attack


def _create_normal_windows(syscalls: List[str]) -> List[List[str]]:
    """Create windows for normal traces with fallback for shorter traces."""
    window_size, stride = 1024, 512
    windows = []
    
    for start in range(0, len(syscalls) - window_size + 1, stride):
        windows.append(syscalls[start:start + window_size])
    
    if not windows:
        length = len(syscalls)
        if length >= 512:
            windows.append(syscalls[:512])
        elif length >= 256:
            windows.append(syscalls[:256])
        elif length >= 100:
            windows.append(syscalls)
    
    return windows[:50]  # Max 50 windows per normal trace


def create_syscall_windows(syscalls: List[str], is_attack: bool = False) -> List[List[str]]:
    """Create windowed sequences with adaptive sizing."""
    if len(syscalls) < 20:
        return []
    
    if is_attack:
        return _create_attack_windows(syscalls)
    return _create_normal_windows(syscalls)


def save_traces_to_files(traces: List[List[str]], output_directory: str, is_attack: bool = False) -> None:
    """Save syscall traces to text files with windowing."""
    os.makedirs(output_directory, exist_ok=True)
    file_counter = 0
    total_windows = 0
    
    for trace in traces:
        windows = create_syscall_windows(trace, is_attack)
        total_windows += len(windows)
        
        for window in windows:
            with open(f"{output_directory}/trace_{file_counter}.txt", "w") as f:
                for syscall in window:
                    f.write(f"{syscall}(\n")
            file_counter += 1
    
    label = "Attack" if is_attack else "Normal"
    logging.info(f"{label}: {len(traces)} traces → {total_windows} windows → {file_counter} files")


def extract_h5_traces(h5_path: str) -> List[List[str]]:
    """Extract syscall traces from H5 file."""
    traces = []
    try:
        with h5py.File(h5_path, "r") as file:
            data = file["sequences"]
            for i in range(data.shape[0]):
                trace = [str(int(syscall)) for syscall in data[i] if syscall != 0]
                if len(trace) >= 20:
                    traces.append(trace)
        logging.info(f"Extracted {len(traces)} traces from H5 file")
    except Exception as error:
        logging.error(f"Error processing H5 file {h5_path}: {error}")
    return traces


def process_h5_dataset(normal_h5_path: str, attack_h5_path: str) -> None:
    """Process H5 format datasets."""
    logging.info(f"Processing H5 - Normal: {normal_h5_path}, Attack: {attack_h5_path}")
    
    normal_traces = extract_h5_traces(normal_h5_path) if normal_h5_path and os.path.exists(normal_h5_path) else []
    attack_traces = extract_h5_traces(attack_h5_path) if attack_h5_path and os.path.exists(attack_h5_path) else []
    
    # 70/30 split
    def split_dataset(traces: List[List[str]]) -> tuple[List[List[str]], List[List[str]]]:
        split_idx = int(0.7 * len(traces))
        return traces[:split_idx], traces[split_idx:]
    
    train_normal, infer_normal = split_dataset(normal_traces)
    train_attack, infer_attack = split_dataset(attack_traces)
    
    logging.info(f"Split - Train: {len(train_normal)} normal / {len(train_attack)} attack")
    
    # Save traces
    save_traces_to_files(train_normal, "traces_train/normal", is_attack=False)
    save_traces_to_files(train_attack, "traces_train/attack", is_attack=True)
    save_traces_to_files(infer_normal, "traces_infer/normal", is_attack=False)
    save_traces_to_files(infer_attack, "traces_infer/attack", is_attack=True)


def extract_zip_files(dataset_path: str, temp_dir: str) -> tuple[List[Path], dict]:
    """Extract ZIP files and return syscall files and metadata."""
    syscall_files = []
    json_metadata = {}
    
    for zip_path in Path(dataset_path).rglob("*.zip"):
        # Skip unnecessary folder
        if "__MACOSX" in str(zip_path) or ".DS_Store" in str(zip_path):
            continue
            
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                for file_info in zip_file.filelist:
                    # Skip unnecessary file
                    if (".DS_Store" in file_info.filename):
                        continue
                        
                    if file_info.filename.endswith((".sc", ".json")):
                        stem = Path(file_info.filename).stem
                        ext = Path(file_info.filename).suffix
                        extracted_path = Path(temp_dir) / f"{stem}{ext}"
                        extracted_path.write_bytes(zip_file.read(file_info))
                        
                        if ext == ".sc":
                            syscall_files.append(extracted_path)
                        elif ext == ".json":
                            try:
                                with open(extracted_path, "r") as f:
                                    json_metadata[stem] = json.load(f)
                            except Exception:
                                pass
        except Exception as e:
            logging.warning(f"Error processing {zip_path}: {e}")
    
    return syscall_files, json_metadata


def get_attack_timestamp(metadata: dict) -> int | None:
    """Extract attack timestamp from metadata."""
    if "time" not in metadata or "exploit" not in metadata["time"]:
        return None
    
    exploit_events = metadata["time"]["exploit"]
    if not exploit_events or not isinstance(exploit_events, list):
        return None
    
    return int(exploit_events[0].get("absolute", 0))


def parse_syscall_file(syscall_file: Path) -> List[tuple]:
    """Parse syscall file and return list of (timestamp, syscall) tuples."""
    all_syscalls = []
    try:
        with open(syscall_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    timestamp = int(float(parts[0]))
                    syscall = parts[5]
                    all_syscalls.append((timestamp, syscall))
    except Exception:
        pass
    
    return all_syscalls


def extract_traces_from_syscalls(
    all_syscalls: List[tuple], is_exploit: bool, attack_timestamp: int | None
) -> tuple[List[str], List[str]]:
    """Extract normal and attack traces from syscall data."""
    normal_trace: List[str] = []
    attack_trace: List[str] = []
    
    if not is_exploit:
        syscalls = [sc for _, sc in all_syscalls]
        if len(syscalls) >= 20:
            normal_trace = syscalls
        return normal_trace, attack_trace
    
    if not attack_timestamp:
        syscalls = [sc for _, sc in all_syscalls]
        if len(syscalls) >= 20:
            attack_trace = syscalls
        return normal_trace, attack_trace
    
    # Find attack start
    attack_start = next((i for i, (ts, _) in enumerate(all_syscalls) if ts >= attack_timestamp), None)
    
    if attack_start is None:
        return normal_trace, attack_trace
    
    # Attack sequence: from timestamp onwards
    attack_syscalls = [sc for _, sc in all_syscalls[attack_start:]]
    if len(attack_syscalls) >= 20:
        attack_trace = attack_syscalls
    
    # Normal sequence: before attack
    if attack_start > 20:
        normal_syscalls = [sc for _, sc in all_syscalls[:attack_start]]
        if len(normal_syscalls) >= 20:
            normal_trace = normal_syscalls
    
    return normal_trace, attack_trace


def split_dataset(traces: List[List[str]]) -> tuple[List[List[str]], List[List[str]]]:
    """Split dataset into train and inference sets (70/30)."""
    if not traces:
        return [], []
    split_idx = int(0.7 * len(traces))
    return traces[:split_idx], traces[split_idx:]


def process_lidds_dataset(dataset_path: str) -> None:
    """Process LID-DS format dataset with timestamp-based attack extraction."""
    logging.info(f"Processing LID-DS dataset: {dataset_path}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        syscall_files, json_metadata = extract_zip_files(dataset_path, temp_dir)
        
        exploit_count = sum(1 for meta in json_metadata.values() if meta.get("exploit", False))
        logging.info(f"Extracted {len(syscall_files)} .sc files, {exploit_count} exploits")
        
        normal_traces, attack_traces = [], []
        attack_traces_extracted = 0
        
        for syscall_file in syscall_files:
            stem = syscall_file.stem
            metadata = json_metadata.get(stem, {})
            is_exploit = metadata.get("exploit", False)
            
            attack_timestamp = get_attack_timestamp(metadata) if is_exploit else None
            all_syscalls = parse_syscall_file(syscall_file)
            
            if not all_syscalls:
                continue
            
            normal_trace, attack_trace = extract_traces_from_syscalls(all_syscalls, is_exploit, attack_timestamp)
            
            if normal_trace:
                normal_traces.append(normal_trace)
            if attack_trace:
                attack_traces.append(attack_trace)
                attack_traces_extracted += 1
        
        logging.info(f"Extracted {attack_traces_extracted} attacks, {len(normal_traces)} normal traces")
        
        if not normal_traces and not attack_traces:
            logging.error("No valid traces found!")
            return
        
        train_normal, infer_normal = split_dataset(normal_traces)
        train_attack, infer_attack = split_dataset(attack_traces)
        
        logging.info(f"Train: {len(train_normal)} normal / {len(train_attack)} attack")
        logging.info(f"Infer: {len(infer_normal)} normal / {len(infer_attack)} attack")
        
        # Save traces
        save_traces_to_files(train_normal, "traces_train/normal", is_attack=False)
        save_traces_to_files(train_attack, "traces_train/attack", is_attack=True)
        save_traces_to_files(infer_normal, "traces_infer/normal", is_attack=False)
        save_traces_to_files(infer_attack, "traces_infer/attack", is_attack=True)


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
        else:
            logging.error("Both normal and attack H5 files must be specified")
        
        if args.preprocess:
            output_path = preprocess_dataset("traces_train", False, False, PKL_TRACES_FILENAME)
            logging.info(f"Graph preprocessing completed: {output_path}")
    else:
        logging.error("No dataset specified. Use -l for LID-DS or -n/-a for H5 files")


if __name__ == "__main__":
    main()