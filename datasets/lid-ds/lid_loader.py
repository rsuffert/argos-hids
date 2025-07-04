#!/usr/bin/env python3
"""
LID-DS Standalone Loader
Converts LID-DS syscall data directly for ML training.
"""

import os
import logging
import h5py
import numpy as np
import zipfile
import tempfile
import shutil
from typing import Dict, List, Set


def extract_syscalls_from_sc_file(sc_file_path: str) -> List[str]:
    """Extract syscall names from Sysdig/Falco .sc trace file."""
    syscalls = []
    
    try:
        with open(sc_file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    
                    syscall_name = parts[5].strip()
                    if syscall_name:
                        syscalls.append(syscall_name)
                        
                except (IndexError, ValueError) as e:
                    logging.debug(f"Error parsing line {line_num} in {sc_file_path}: {e}")
                    continue
                    
    except Exception as e:
        logging.error(f"Error reading .sc file {sc_file_path}: {e}")
        return []
    
    logging.info(f"Extracted {len(syscalls)} syscalls from {sc_file_path}")
    return syscalls

def convert_syscalls_to_ids(syscalls: List[str], syscall_lookup: Dict[str, int]) -> List[int]:
    """Convert syscall names to their integer IDs."""
    syscall_ids = []
    for syscall_name in syscalls:
        if syscall_name in syscall_lookup:
            syscall_ids.append(syscall_lookup[syscall_name])
        else:
            syscall_ids.append(0)  # Unknown syscall
            logging.warning(f"Unknown syscall: {syscall_name}")
    
    return syscall_ids

def append_seq_to_h5(sequence: List[int], h5_path: str) -> None:
    """Append a sequence of integers to an HDF5 file."""
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

def create_syscall_table_from_mappings(syscall_mappings: Dict[str, int], output_path: str) -> None:
    """Create a syscall table file from LID-DS syscall mappings."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Syscall table generated from LID-DS scenario data\n")
        f.write("# Format: ID ABI NAME\n")
        f.write("#\n")
        
        f.write("0\tcommon\tUNKNOWN\n")
        
        for syscall_name, syscall_id in sorted(syscall_mappings.items(), key=lambda x: x[1]):
            f.write(f"{syscall_id}\tcommon\t{syscall_name}\n")
    
    logging.info(f"Created syscall table with {len(syscall_mappings)} syscalls at: {output_path}")

def extract_syscalls_from_zip_file(zip_path: str) -> Set[str]:
    """Extract unique syscalls from a zip file temporarily."""
    all_syscalls = set()
    temp_dir = None
    
    try:
        temp_dir = tempfile.mkdtemp(prefix="lid_ds_syscall_")
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        
        for root, _dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".sc"):
                    sc_file_path = os.path.join(root, file)
                    syscalls = extract_syscalls_from_sc_file(sc_file_path)
                    all_syscalls.update(syscalls)
        
    except Exception as e:
        logging.error(f"Error extracting syscalls from zip {zip_path}: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return all_syscalls

def process_zip_temporarily(zip_path: str, syscall_mappings: Dict[str, int], h5_output_path: str) -> int:
    """Process a zip file temporarily to save disk space."""
    sequence_count = 0
    temp_dir = None
    
    try:
        temp_dir = tempfile.mkdtemp(prefix="lid_ds_temp_")
        logging.info(f"Extracting {os.path.basename(zip_path)} to temp dir: {temp_dir}")
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        
        for root, _dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".sc"):
                    sc_file_path = os.path.join(root, file)
                    
                    syscalls = extract_syscalls_from_sc_file(sc_file_path)
                    if syscalls:
                        syscall_ids = convert_syscalls_to_ids(syscalls, syscall_mappings)
                        append_seq_to_h5(syscall_ids, h5_output_path)
                        sequence_count += 1
                        
                        relative_path = os.path.relpath(sc_file_path, temp_dir)
                        logging.info(f"Processed sequence from zip: {relative_path} ({len(syscalls)} syscalls)")
        
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile) as e:
        logging.error(f"Operational error while processing zip {zip_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while processing zip {zip_path}: {e}")
        raise
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temp dir: {temp_dir}")
    
    return sequence_count

def discover_all_scenarios(base_path: str = "SCENARIOS") -> List[str]:
    """Discover all available LID-DS scenarios."""
    scenarios: List[str] = []
    
    if not os.path.exists(base_path):
        logging.warning(f"Scenarios directory not found: {base_path}")
        return scenarios
    
    for item in os.listdir(base_path):
        scenario_path = os.path.join(base_path, item)
        if os.path.isdir(scenario_path):
            training_path = os.path.join(scenario_path, "training")
            test_path = os.path.join(scenario_path, "test")
            
            if os.path.exists(training_path) or os.path.exists(test_path):
                scenarios.append(scenario_path)
                logging.info(f"Found scenario: {item}")
    
    return scenarios

def extract_syscalls_from_scenario_files(scenario_path: str) -> tuple[Set[str], int]:
    """Extract all unique syscalls from a scenario's files."""
    all_syscalls = set()
    file_count = 0
    
    for root, _dirs, files in os.walk(scenario_path):
        for file in files:
            if file.endswith(".sc"):
                sc_file_path = os.path.join(root, file)
                file_count += 1
                syscalls = extract_syscalls_from_sc_file(sc_file_path)
                all_syscalls.update(syscalls)
            
            elif file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                zip_syscalls = extract_syscalls_from_zip_file(zip_path)
                all_syscalls.update(zip_syscalls)
                if zip_syscalls:
                    file_count += 1
    
    return all_syscalls, file_count

def get_file_list_from_directory(directory_path: str) -> Set[str]:
    """Get list of file/directory names from a directory."""
    file_set: Set[str] = set()
    
    if not os.path.exists(directory_path):
        logging.warning(f"Directory does not exist: {directory_path}")
        return file_set
    
    for item in os.listdir(directory_path):
        file_set.add(item)
        logging.debug(f"Found normal file: {item}")
    
    logging.info(f"Found {len(file_set)} normal files to exclude")
    return file_set

def process_scenario_data_by_type(
    scenario_path: str,
    subdir: str,
    data_type: str,
    syscall_mappings: Dict[str, int],
    h5_output_path: str
) -> int:
    """Process LID-DS scenario data of a specific type and save to HDF5."""
    data_path = os.path.join(scenario_path, subdir)
    sequence_count = 0
    
    if not os.path.exists(data_path):
        logging.warning(f"Data path does not exist: {data_path}")
        return sequence_count
    
    logging.info(f"Processing {data_type} data from: {data_path}")
    
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        
        # Handle direct .sc files
        if item.endswith(".sc"):
            logging.info(f"Processing direct .sc file: {item}")
            try:
                syscalls = extract_syscalls_from_sc_file(item_path)
                if syscalls:
                    syscall_ids = convert_syscalls_to_ids(syscalls, syscall_mappings)
                    append_seq_to_h5(syscall_ids, h5_output_path)
                    sequence_count += 1
                    logging.info(f"Processed {data_type} .sc file: {item} ({len(syscalls)} syscalls)")
                else:
                    logging.warning(f"No syscalls found in .sc file: {item}")
            except Exception as e:
                logging.error(f"Error processing .sc file {item}: {e}")
        
        # Handle directories
        elif os.path.isdir(item_path):
            sc_file = os.path.join(item_path, f"{item}.sc")
            if os.path.exists(sc_file):
                syscalls = extract_syscalls_from_sc_file(sc_file)
                if syscalls:
                    syscall_ids = convert_syscalls_to_ids(syscalls, syscall_mappings)
                    append_seq_to_h5(syscall_ids, h5_output_path)
                    sequence_count += 1
                    logging.info(f"Processed {data_type} sequence: {item} ({len(syscalls)} syscalls)")
        
        # Handle ZIP files
        elif item.endswith(".zip"):
            zip_sequences = process_zip_temporarily(item_path, syscall_mappings, h5_output_path)
            sequence_count += zip_sequences
            logging.info(f"Processed zip file: {item} ({zip_sequences} sequences)")
    
    return sequence_count

def _process_attack_item(item: str, item_path: str, syscall_mappings: Dict[str, int], h5_output_path: str) -> int:
    """Helper to process a single attack item (file, dir, or zip)."""
    sequence_count = 0
    # Handle direct .sc files
    if item.endswith(".sc"):
        try:
            syscalls = extract_syscalls_from_sc_file(item_path)
            if syscalls:
                syscall_ids = convert_syscalls_to_ids(syscalls, syscall_mappings)
                append_seq_to_h5(syscall_ids, h5_output_path)
                sequence_count += 1
                logging.info(f"Processed ATTACK .sc file: {item} ({len(syscalls)} syscalls)")
        except Exception as e:
            logging.error(f"Error processing attack .sc file {item}: {e}")
    # Handle directories
    elif os.path.isdir(item_path):
        sc_file = os.path.join(item_path, f"{item}.sc")
        if os.path.exists(sc_file):
            syscalls = extract_syscalls_from_sc_file(sc_file)
            if syscalls:
                syscall_ids = convert_syscalls_to_ids(syscalls, syscall_mappings)
                append_seq_to_h5(syscall_ids, h5_output_path)
                sequence_count += 1
                logging.info(f"Processed ATTACK sequence: {item} ({len(syscalls)} syscalls)")
    # Handle ZIP files
    elif item.endswith(".zip"):
        zip_sequences = process_zip_temporarily(item_path, syscall_mappings, h5_output_path)
        sequence_count += zip_sequences
        logging.info(f"Processed attack zip file: {item} ({zip_sequences} sequences)")
    return sequence_count

def process_mixed_data_with_exclusions(
    scenario_path: str,
    subdir: str,
    exclude_files: Set[str],
    syscall_mappings: Dict[str, int],
    h5_output_path: str
) -> int:
    """Process mixed data directory, excluding specified files."""
    data_path = os.path.join(scenario_path, subdir)
    sequence_count = 0

    if not os.path.exists(data_path):
        logging.warning(f"Data path does not exist: {data_path}")
        return sequence_count

    logging.info(f"Processing mixed data from: {data_path}")
    logging.info(f"Excluding {len(exclude_files)} normal files")

    for item in os.listdir(data_path):
        if item in exclude_files:
            logging.debug(f"Skipping normal file: {item}")
            continue

        item_path = os.path.join(data_path, item)
        sequence_count += _process_attack_item(item, item_path, syscall_mappings, h5_output_path)

    return sequence_count

# Main processing function
def process_all_scenarios_hdf5_separated(
    output_dir: str,
    scenarios_base_path: str = "SCENARIOS"
) -> None:
    """
    Process all scenarios using exclusion logic,
    filter repeated data to distinct attack from normal.
    """
    logging.info("Starting automatic LID-DS scenarios filtering data...")
    
    scenarios = discover_all_scenarios(scenarios_base_path)
    if not scenarios:
        logging.error("No scenarios found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract syscalls from all scenarios
    all_syscalls = set()
    total_files = 0
    
    for scenario_path in scenarios:
        scenario_syscalls, file_count = extract_syscalls_from_scenario_files(scenario_path)
        all_syscalls.update(scenario_syscalls)
        total_files += file_count
        scenario_name = os.path.basename(scenario_path)
        logging.info(f"Scenario {scenario_name}: {len(scenario_syscalls)} syscalls, {file_count} files")
    
    # Create syscall mappings
    unique_syscalls = sorted(list(all_syscalls))
    syscall_mappings = {}
    for i, syscall_name in enumerate(unique_syscalls, 1):
        syscall_mappings[syscall_name] = i
    
    # Create output files
    syscall_table_path = os.path.join(output_dir, "syscall_64.tbl")
    normal_h5_path = os.path.join(output_dir, "0_normal.h5")
    attack_h5_path = os.path.join(output_dir, "1_attack.h5")
    
    create_syscall_table_from_mappings(syscall_mappings, syscall_table_path)
    
    # Remove existing files
    for h5_path in [normal_h5_path, attack_h5_path]:
        if os.path.exists(h5_path):
            os.remove(h5_path)
    
    # Process all scenarios
    total_normal = 0
    total_attack = 0
    
    for scenario_path in scenarios:
        scenario_name = os.path.basename(scenario_path)
        logging.info(f"Processing scenario: {scenario_name}")
        
        # Normal data
        training_normal = process_scenario_data_by_type(
            scenario_path, "training", "normal", syscall_mappings, normal_h5_path
        )
        
        test_normal = process_scenario_data_by_type(
            scenario_path, "test/normal", "normal", syscall_mappings, normal_h5_path
        )
        
        # Attack data with exclusions
        normal_files = get_file_list_from_directory(
            os.path.join(scenario_path, "test", "normal")
        )
        
        attack_count = process_mixed_data_with_exclusions(
            scenario_path, "test/normal_and_attack", normal_files,
            syscall_mappings, attack_h5_path
        )
        
        scenario_normal = training_normal + test_normal
        total_normal += scenario_normal
        total_attack += attack_count
        
        logging.info(f"Scenario {scenario_name}: {scenario_normal} normal, {attack_count} attack")
    
    # Print summary
    print("\n" + "="*60)
    print("LID-DS CONVERSION COMPLETE")
    print("="*60)
    print(f"Normal data: {normal_h5_path} ({total_normal} sequences)")
    print(f"Attack data: {attack_h5_path} ({total_attack} sequences)")
    print(f"Syscall table: {syscall_table_path} ({len(unique_syscalls)} syscalls)")

def process_all_scenarios_hdf5_val_folder(
    output_dir: str,
    scenarios_base_path: str = "SCENARIOS"
) -> None:
    """Process all scenarios using test folders for training and validation folder for validation."""
    logging.info("Starting LID-DS scenarios processing with validation folder...")
    
    scenarios = discover_all_scenarios(scenarios_base_path)
    if not scenarios:
        logging.error("No scenarios found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract syscalls from all scenarios
    all_syscalls = set()
    total_files = 0
    
    for scenario_path in scenarios:
        scenario_syscalls, file_count = extract_syscalls_from_scenario_files(scenario_path)
        all_syscalls.update(scenario_syscalls)
        total_files += file_count
        scenario_name = os.path.basename(scenario_path)
        logging.info(f"Scenario {scenario_name}: {len(scenario_syscalls)} syscalls, {file_count} files")
    
    # Create syscall mappings
    unique_syscalls = sorted(list(all_syscalls))
    syscall_mappings = {}
    for i, syscall_name in enumerate(unique_syscalls, 1):
        syscall_mappings[syscall_name] = i
    
    # Create output files
    syscall_table_path = os.path.join(output_dir, "syscall_64.tbl")
    normal_h5_path = os.path.join(output_dir, "0_normal.h5")
    attack_h5_path = os.path.join(output_dir, "1_attack.h5")
    validation_h5_path = os.path.join(output_dir, "validation.h5")
    
    create_syscall_table_from_mappings(syscall_mappings, syscall_table_path)
    
    # Remove existing files
    for h5_path in [normal_h5_path, attack_h5_path, validation_h5_path]:
        if os.path.exists(h5_path):
            os.remove(h5_path)
    
    # Process all scenarios
    total_normal = 0
    total_attack = 0
    total_validation = 0
    
    for scenario_path in scenarios:
        scenario_name = os.path.basename(scenario_path)
        logging.info(f"Processing scenario: {scenario_name}")
        
        # TRAINING DATA: Normal data from test/normal
        test_normal = process_scenario_data_by_type(
            scenario_path, "test/normal", "normal", syscall_mappings, normal_h5_path
        )
        
        # TRAINING DATA: Attack data from test/normal_and_attack (excluding normal files)
        normal_files = get_file_list_from_directory(
            os.path.join(scenario_path, "test", "normal")
        )
        
        attack_count = process_mixed_data_with_exclusions(
            scenario_path, "test/normal_and_attack", normal_files,
            syscall_mappings, attack_h5_path
        )
        
        # VALIDATION DATA: All data from validation folder
        validation_count = process_scenario_data_by_type(
            scenario_path, "validation", "validation", syscall_mappings, validation_h5_path
        )
        
        total_normal += test_normal
        total_attack += attack_count
        total_validation += validation_count
        
        logging.info(
            f"Scenario {scenario_name}: Train({test_normal} normal, {attack_count} attack), "
            f"Val({validation_count} mixed)"
        )
    
    # Print summary
    print("\n" + "="*60)
    print("LID-DS CONVERSION COMPLETE")
    print("="*60)
    print("TRAINING DATA:")
    print(f"  Normal: {normal_h5_path} ({total_normal} sequences)")
    print(f"  Attack: {attack_h5_path} ({total_attack} sequences)")
    print("VALIDATION DATA:")
    print(f"  Mixed: {validation_h5_path} ({total_validation} sequences)")
    print(f"Syscall table: {syscall_table_path} ({len(unique_syscalls)} syscalls)")
    print(
        f"\nTraining class balance: {total_normal} normal : {total_attack} attack = "
        f"{total_normal/total_attack:.1f}:1"
    )

# Main execution
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    scenarios_base = os.path.join(os.path.dirname(__file__), "SCENARIOS")
    output_dir = os.path.join(os.path.dirname(__file__), "processed_lid_data")
    
    assert os.path.exists(scenarios_base), f"Scenarios directory does not exist: {scenarios_base}"
    
    scenarios = discover_all_scenarios(scenarios_base)
    assert scenarios, "No scenarios found in SCENARIOS directory."
    
    print("=" * 60)
    print("LID-DS LOADER")
    print("=" * 60)
    print(f"Found {len(scenarios)} scenarios:")
    for scenario in scenarios:
        print(f"  - {os.path.basename(scenario)}")
    
    process_all_scenarios_hdf5_val_folder(output_dir, scenarios_base)
    
    print("\nFiles ready for ML training:")
    print("  - 0_normal.h5 (labeled normal sequences for training)")
    print("  - 1_attack.h5 (labeled attack sequences for training)")
    print("  - validation.h5 (mixed validation sequences)")
    print("  - syscall_64.tbl (syscall vocabulary)")
