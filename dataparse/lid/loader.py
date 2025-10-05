"""LID Dataset Loader for ARGOS-HIDS."""

import os
import json
import logging
import argparse
import zipfile
import tempfile
import h5py
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from preprocessing.graph_preprocess_dataset import preprocess_dataset # type: ignore
except ImportError:
    preprocess_dataset = None

PKL_TRACES_FILENAME = "processed_graphs.pkl"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Syscall Classification Pipeline")
    parser.add_argument("-l", "--lidds", type=str, help="Path to LID-DS dataset directory")
    parser.add_argument("-n", "--normal", type=str, help="Path to normal traces H5 file")
    parser.add_argument("-a", "--attack", type=str, help="Path to attack traces H5 file")
    parser.add_argument("-p", "--preprocess", action="store_true", help="Preprocess training data to graphs")
    return parser.parse_args()


def create_windows(syscalls: List[str], max_windows: int = 50) -> List[List[str]]:
    """Create windowed sequences."""
    if len(syscalls) < 20:
        return []
    
    window_size = min(1024, len(syscalls))
    stride = window_size // 2
    
    windows = []
    for i in range(0, len(syscalls) - window_size + 1, stride):
        windows.append(syscalls[i:i + window_size])
        if len(windows) >= max_windows:
            break
    
    return windows or [syscalls]


def save_traces(traces: List[List[str]], output_dir: str, is_attack: bool = False) -> None:
    """Save traces to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    file_count = 0
    max_windows = 10 if is_attack else 50
    
    for trace in traces:
        for window in create_windows(trace, max_windows):
            file_path = f"{output_dir}/trace_{file_count}.txt"
            with open(file_path, "w") as f:
                f.write("\n".join(f"{sc}(" for sc in window))
            file_count += 1
    
    label = "Attack" if is_attack else "Normal"
    logging.info(f"{label}: {len(traces)} traces → {file_count} files")


def load_h5_data(h5_path: str) -> List[List[str]]:
    """Load syscall traces from H5 file."""
    if not h5_path or not os.path.exists(h5_path):
        return []
    
    try:
        with h5py.File(h5_path, "r") as f:
            data = f["sequences"]
            traces = []
            for i in range(data.shape[0]):
                trace = [str(int(s)) for s in data[i] if s != 0]
                if len(trace) >= 20:
                    traces.append(trace)
            return traces
    except Exception as e:
        logging.error(f"Error loading {h5_path}: {e}")
        return []


def split_data(traces: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    """Split data 70/30."""
    if not traces:
        return [], []
    split_idx = int(0.7 * len(traces))
    return traces[:split_idx], traces[split_idx:]


def process_h5_format(normal_path: str, attack_path: str) -> None:
    """Process H5 format datasets."""
    logging.info("Processing H5 datasets")
    
    normal_traces = load_h5_data(normal_path)
    attack_traces = load_h5_data(attack_path)
    
    train_normal, test_normal = split_data(normal_traces)
    train_attack, test_attack = split_data(attack_traces)
    
    # Save all traces
    datasets = [
        (train_normal, "traces_train/normal", False),
        (train_attack, "traces_train/attack", True),
        (test_normal, "traces_infer/normal", False),
        (test_attack, "traces_infer/attack", True)
    ]
    
    for traces, path, is_attack in datasets:
        save_traces(traces, path, is_attack)


def should_skip(path: str) -> bool:
    """Check if file should be skipped."""
    return "__MACOSX" in path or ".DS_Store" in path


def extract_files(dataset_path: str, temp_dir: str) -> Tuple[List[Path], dict]:
    """Extract files from ZIP archives."""
    syscall_files = []
    metadata = {}
    
    for zip_path in Path(dataset_path).rglob("*.zip"):
        if should_skip(str(zip_path)):
            continue
        
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.filelist:
                    if should_skip(info.filename):
                        continue
                    
                    if info.filename.endswith((".sc", ".json")):
                        name = Path(info.filename).stem
                        ext = Path(info.filename).suffix
                        out_path = Path(temp_dir) / f"{name}{ext}"
                        out_path.write_bytes(zf.read(info))
                        
                        if ext == ".sc":
                            syscall_files.append(out_path)
                        elif ext == ".json":
                            try:
                                with open(out_path) as f:
                                    metadata[name] = json.load(f)
                            except Exception:
                                pass
        except Exception as e:
            logging.warning(f"Error with {zip_path}: {e}")
    
    return syscall_files, metadata


def parse_syscalls(file_path: Path) -> List[Tuple[int, str]]:
    """Parse syscall file."""
    try:
        with open(file_path) as f:
            result = []
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    ts = int(parts[0]) if len(parts) >= 2 else line_num
                    sc = parts[1] if len(parts) >= 2 else parts[0]
                except ValueError:
                    ts, sc = line_num, parts[0]
                
                result.append((ts, sc))
            return result
    except Exception as e:
        logging.warning(f"Error parsing {file_path}: {e}")
        return []


def extract_traces(
    syscalls: List[Tuple[int, str]], is_exploit: bool, attack_time: Optional[int]
) -> Tuple[List[str], List[str]]:
    """Extract normal and attack traces."""
    names = [sc for _, sc in syscalls]
    
    if len(names) < 20:
        return [], []
    
    if not is_exploit:
        return names, []
    
    if not attack_time:
        return [], names
    
    return _split_at_attack_time(syscalls, attack_time)


def _split_at_attack_time(syscalls: List[Tuple[int, str]], attack_time: int) -> Tuple[List[str], List[str]]:
    """Split syscalls at attack timestamp."""
    attack_idx = _find_attack_index(syscalls, attack_time)
    
    normal = [sc for _, sc in syscalls[:attack_idx]] if attack_idx > 20 else []
    attack = [sc for _, sc in syscalls[attack_idx:]]
    
    return (
        normal if len(normal) >= 20 else [],
        attack if len(attack) >= 20 else []
    )


def _find_attack_index(syscalls: List[Tuple[int, str]], attack_time: int) -> int:
    """Find index where attack starts."""
    for i, (ts, _) in enumerate(syscalls):
        if ts >= attack_time:
            return i
    return len(syscalls)


def process_lidds_format(dataset_path: str) -> None:
    """Process LID-DS format dataset."""
    logging.info(f"Processing LID-DS: {dataset_path}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        syscall_files, metadata = extract_files(dataset_path, temp_dir)
        
        logging.info(f"Found {len(syscall_files)} syscall files")
        
        normal_traces = []
        attack_traces = []
        
        for sc_file in syscall_files:
            name = sc_file.stem
            meta = metadata.get(name, {})
            is_exploit = meta.get("exploit", False)
            attack_time = meta.get("attack_timestamp") if is_exploit else None
            
            syscalls = parse_syscalls(sc_file)
            if not syscalls:
                continue
            
            normal, attack = extract_traces(syscalls, is_exploit, attack_time)
            
            if normal:
                normal_traces.append(normal)
            if attack:
                attack_traces.append(attack)
        
        logging.info(f"Extracted {len(normal_traces)} normal, {len(attack_traces)} attack traces")
        
        if not normal_traces and not attack_traces:
            logging.error("No valid traces found!")
            return
        
        train_normal, test_normal = split_data(normal_traces)
        train_attack, test_attack = split_data(attack_traces)
        
        # Save all traces
        datasets = [
            (train_normal, "traces_train/normal", False),
            (train_attack, "traces_train/attack", True),
            (test_normal, "traces_infer/normal", False),
            (test_attack, "traces_infer/attack", True)
        ]
        
        for traces, path, is_attack in datasets:
            save_traces(traces, path, is_attack)


def main() -> None:
    """Main function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    args = parse_arguments()
    
    if args.lidds:
        _handle_lidds_processing(args)
    elif args.normal or args.attack:
        _handle_h5_processing(args)
    else:
        logging.error("No dataset specified. Use -l for LID-DS or -n/-a for H5")


def _handle_lidds_processing(args: argparse.Namespace) -> None:
    """Handle LID-DS format processing."""
    process_lidds_format(args.lidds)
    
    if args.preprocess and preprocess_dataset:
        output = preprocess_dataset("traces_train", False, False, PKL_TRACES_FILENAME)
        logging.info(f"Preprocessing complete: {output}")


def _handle_h5_processing(args: argparse.Namespace) -> None:
    """Handle H5 format processing."""
    normal_h5 = args.normal or os.getenv("LID_NORMAL")
    attack_h5 = args.attack or os.getenv("LID_ATTACK")
    
    if not normal_h5 or not attack_h5:
        logging.error("Need both normal and attack H5 files")
        return
    
    process_h5_format(normal_h5, attack_h5)
    
    if args.preprocess and preprocess_dataset:
        output = preprocess_dataset("traces_train", False, False, PKL_TRACES_FILENAME)
        logging.info(f"Preprocessing complete: {output}")