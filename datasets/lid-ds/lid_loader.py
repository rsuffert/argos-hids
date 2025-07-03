#!/usr/bin/env python3
"""
LID-DS Standalone Loader
Converts LID-DS syscall data directly for the ML training.
"""

import os
import logging
import h5py
import numpy as np
import zipfile
import tempfile
import shutil
from typing import Dict, List, Optional, Set

def process_zip_temporarily(zip_path: str, syscall_mappings: Dict[str, int], h5_output_path: str) -> int:
    """
    Process a zip file temporarily to save disk space - extract, process, then clean up.
    
    Args:
        zip_path: Path to the zip file
        syscall_mappings: Dictionary mapping syscall names to IDs
        h5_output_path: Path to output HDF5 file
        
    Returns:
        Number of sequences processed
    """
    sequence_count = 0
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="lid_ds_temp_")
        logging.info(f"Extracting {os.path.basename(zip_path)} to temp dir: {temp_dir}")
        
        # Extract zip to temporary directory
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find and process .sc files in the extracted content
        for root, _dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".sc"):
                    sc_file_path = os.path.join(root, file)
                    
                    # Extract syscalls from .sc file
                    syscalls = extract_syscalls_from_sc_file(sc_file_path)
                    if syscalls:
                        # Convert syscall names to IDs
                        syscall_ids = convert_syscalls_to_ids(syscalls, syscall_mappings)
                        
                        # Append to HDF5 file
                        append_seq_to_h5(syscall_ids, h5_output_path)
                        sequence_count += 1
                        
                        # Get a readable name from the file path
                        relative_path = os.path.relpath(sc_file_path, temp_dir)
                        logging.info(f"Processed sequence from zip: {relative_path} ({len(syscalls)} syscalls)")
        
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile) as e:
        logging.error(f"Operational error while processing zip {zip_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while processing zip {zip_path}: {e}")
        raise
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temp dir: {temp_dir}")
    
    return sequence_count


def discover_all_scenarios(base_path: str = "SCENARIOS") -> List[str]:
    """
    Discover all available LID-DS scenarios.
    
    Args:
        base_path: Base path containing scenario directories
        
    Returns:
        List of scenario directory paths
    """
    scenarios:List[str] = []
    
    if not os.path.exists(base_path):
        logging.warning(f"Scenarios directory not found: {base_path}")
        return scenarios
    
    for item in os.listdir(base_path):
        scenario_path = os.path.join(base_path, item)
        if os.path.isdir(scenario_path):
            # Check if it looks like a scenario (has training or test directories)
            training_path = os.path.join(scenario_path, "training")
            test_path = os.path.join(scenario_path, "test")
            
            if os.path.exists(training_path) or os.path.exists(test_path):
                scenarios.append(scenario_path)
                logging.info(f"Found scenario: {item}")
    
    return scenarios


def process_all_scenarios_to_hdf5(
    output_dir: str,
    scenarios_base_path: str = "SCENARIOS"
) -> None:
    """
    Automatically discover and process all LID-DS scenarios to HDF5 format.
    
    Args:
        output_dir: Output directory for HDF5 files and syscall table
        scenarios_base_path: Base path containing scenario directories
    """
    logging.info("Starting automatic LID-DS scenarios processing...")
    logging.info(f"Searching for scenarios in: {scenarios_base_path}")
    logging.info(f"Output: {output_dir}")
    
    # Discover all scenarios
    scenarios = discover_all_scenarios(scenarios_base_path)
    
    if not scenarios:
        logging.error("No scenarios found!")
        return
    
    logging.info(f"Found {len(scenarios)} scenarios: {[os.path.basename(s) for s in scenarios]}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract syscalls from all scenarios combined
    logging.info("Extracting syscalls from all scenarios...")
    all_syscalls = set()
    total_files = 0
    
    for scenario_path in scenarios:
        scenario_syscalls, file_count = extract_syscalls_from_scenario_files(scenario_path)
        all_syscalls.update(scenario_syscalls)
        total_files += file_count
        scenario_name = os.path.basename(scenario_path)
        logging.info(f"Scenario {scenario_name}: {len(scenario_syscalls)} syscalls, {file_count} files")
    
    # Create unified syscall mappings
    unique_syscalls = sorted(list(all_syscalls))
    syscall_mappings = {}
    for i, syscall_name in enumerate(unique_syscalls, 1):
        syscall_mappings[syscall_name] = i
    
    logging.info(f"Total unique syscalls across all scenarios: {len(unique_syscalls)}")
    logging.info(f"Total files processed: {total_files}")
    
    # Create syscall table file
    syscall_table_path = os.path.join(output_dir, "syscall_64.tbl")
    create_syscall_table_from_mappings(syscall_mappings, syscall_table_path)
    
    # HDF5 output files
    normal_h5_path = os.path.join(output_dir, "0_normal.h5")
    attack_h5_path = os.path.join(output_dir, "1_attack.h5")
    
    # Remove existing HDF5 files to start fresh
    for h5_path in [normal_h5_path, attack_h5_path]:
        if os.path.exists(h5_path):
            os.remove(h5_path)
            logging.info(f"Removed existing file: {h5_path}")
    
    # Process all scenarios - both normal and attack data
    total_normal = 0
    total_attack = 0
    
    for scenario_path in scenarios:
        scenario_name = os.path.basename(scenario_path)
        logging.info(f"Processing scenario: {scenario_name}")
        
        # Process normal data (training)
        normal_count = process_scenario_data_by_type(
            scenario_path, "training", "normal", syscall_mappings, normal_h5_path
        )
        
        # Process attack data (test/normal_and_attack)
        attack_count = process_scenario_data_by_type(
            scenario_path, "test/normal_and_attack", "attack", syscall_mappings, attack_h5_path
        )
        
        total_normal += normal_count
        total_attack += attack_count
        
        logging.info(f"Scenario {scenario_name}: {normal_count} normal, {attack_count} attack")
    
    logging.info("All scenarios processing completed!")
    logging.info(f"Total normal sequences: {total_normal}")
    logging.info(f"Total attack sequences: {total_attack}")
    logging.info(f"Syscall table: {syscall_table_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("LID-DS ALL SCENARIOS HDF5 CONVERSION COMPLETE")
    print("="*60)
    print(f"Processed {len(scenarios)} scenarios:")
    for scenario_path in scenarios:
        print(f"  - {os.path.basename(scenario_path)}")
    print("\nFiles ready for ML training:")
    print(f"Normal data: {normal_h5_path} ({total_normal} sequences)")
    print(f"Attack data: {attack_h5_path} ({total_attack} sequences)")
    print(f"Syscall table: {syscall_table_path} ({len(unique_syscalls)} syscalls)")
    print("Update your trainer.py to use these paths:")
    print(f'normal_dataset = H5LazyDataset("{normal_h5_path}", 0)')
    print(f'attack_dataset = H5LazyDataset("{attack_h5_path}", 1)')


def extract_syscalls_from_scenario_files(scenario_path: str) -> tuple[Set[str], int]:
    """
    Extract all unique syscalls from a scenario's files (including zip files).
    
    Args:
        scenario_path: Path to scenario directory
        
    Returns:
        Tuple of (set of unique syscalls, file count)
    """
    all_syscalls = set()
    file_count = 0
    
    # Search for all .sc files in the scenario
    for root, _dirs, files in os.walk(scenario_path):
        for file in files:
            if file.endswith(".sc"):
                sc_file_path = os.path.join(root, file)
                file_count += 1
                
                syscalls = extract_syscalls_from_sc_file(sc_file_path)  # This returns List[str] - GOOD!
                all_syscalls.update(syscalls)  # Add to vocabulary set
            
            elif file.endswith(".zip"):
                # Process zip file temporarily to extract syscalls
                zip_path = os.path.join(root, file)
                # Returns Set[str], which is expected for vocabulary building.
                zip_syscalls = extract_syscalls_from_zip_file(zip_path)  
                all_syscalls.update(zip_syscalls)
                if zip_syscalls:
                    file_count += 1
    
    return all_syscalls, file_count


def extract_syscalls_from_zip_file(zip_path: str) -> Set[str]:
    """
    Extract unique syscalls from a zip file temporarily.
    
    Args:
        zip_path: Path to the zip file
        
    Returns:
        Set of unique syscalls found in the zip (for vocabulary building only)
    """
    all_syscalls = set()
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="lid_ds_syscall_")
        
        # Extract zip to temporary directory
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find and process .sc files in the extracted content
        for root, _dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".sc"):
                    sc_file_path = os.path.join(root, file)
                    syscalls = extract_syscalls_from_sc_file(sc_file_path)  # Returns List[str], order kept
                    all_syscalls.update(syscalls)  # Add to vocabulary set(unique syscalls only for the list)
        
    except Exception as e:
        logging.error(f"Error extracting syscalls from zip {zip_path}: {e}")
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return all_syscalls


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

# Default ABI for syscall table generation
DEFAULT_ABI = "common"

def create_syscall_table_from_mappings(syscall_mappings: Dict[str, int], output_path: str) -> None:
    """
    Create a syscall table file compatible with the loader from LID-DS syscall mappings.
    
    Args:
        syscall_mappings: Dictionary mapping syscall names to IDs
        output_path: Path where to save the syscall table file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Syscall table generated from LID-DS scenario data\n")
        f.write("# Format: ID ABI NAME\n")
        f.write("#\n")
        
        # Add UNKNOWN syscall at ID 0
        f.write(f"0\t{DEFAULT_ABI}\tUNKNOWN\n")
        
        # Add all extracted syscalls
        for syscall_name, syscall_id in sorted(syscall_mappings.items(), key=lambda x: x[1]):
            f.write(f"{syscall_id}\t{DEFAULT_ABI}\t{syscall_name}\n")
    
    logging.info(f"Created syscall table with {len(syscall_mappings)} syscalls at: {output_path}")


def extract_syscalls_from_sc_file(sc_file_path: str) -> List[str]:
    """
    Extract syscalls from a .sc file (LID-DS format).
    
    Args:
        sc_file_path: Path to the .sc file
        
    Returns:
        List of syscall names in order
    """
    syscalls = []
    
    try:
        with open(sc_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    syscall_name = parts[5]  # syscall name is at index 5
                    syscalls.append(syscall_name)
    except Exception as e:
        logging.error(f"Error reading {sc_file_path}: {e}")
    
    return syscalls


def extract_all_syscalls_from_scenario(scenario_path: str) -> Dict[str, int]:
    """
    Extract all unique syscalls from a LID-DS scenario and create mappings.
    
    Args:
        scenario_path: Path to LID-DS scenario directory
        
    Returns:
        Dictionary mapping syscall names to IDs (starting from 1, 0 reserved for UNKNOWN)
    """
    logging.info(f"Extracting syscalls from scenario: {scenario_path}")
    
    all_syscalls = set()
    file_count = 0
    
    # Search for all .sc files in the scenario
    for root, _dirs, files in os.walk(scenario_path):
        for file in files:
            if file.endswith(".sc"):
                sc_file_path = os.path.join(root, file)
                file_count += 1
                logging.info(f"Processing {file_count}: {file}")
                # This returns List[str], with the order preserved.
                syscalls = extract_syscalls_from_sc_file(sc_file_path) 
                all_syscalls.update(syscalls)  # Add to vocabulary set
    
    # Create mappings (alphabetically sorted for consistency)
    unique_syscalls = sorted(list(all_syscalls))
    syscall_mappings = {}
    
    for i, syscall_name in enumerate(unique_syscalls, 1):
        syscall_mappings[syscall_name] = i
    
    logging.info(f"Found {len(unique_syscalls)} unique syscalls across {file_count} files")
    logging.info(f"Syscalls: {', '.join(unique_syscalls[:10])}{'...' if len(unique_syscalls) > 10 else ''}")
    
    return syscall_mappings


def convert_syscalls_to_ids(syscalls: List[str], syscall_lookup: Dict[str, int]) -> List[int]:
    """
    Convert syscall names to their integer IDs.
    
    Args:
        syscalls: List of syscall names
        syscall_lookup: Dictionary mapping syscall names to IDs
        
    Returns:
        List of syscall IDs
    """
    syscall_ids = []
    for syscall_name in syscalls:
        if syscall_name in syscall_lookup:
            syscall_ids.append(syscall_lookup[syscall_name])
        else:
            # Use ID 0 for unknown syscalls
            syscall_ids.append(0)
            logging.warning(f"Unknown syscall: {syscall_name}")
    
    return syscall_ids


def process_scenario_to_hdf5(
    scenario_path: str,
    output_dir: str,
    syscall_mappings: Optional[Dict[str, int]] = None
) -> None:
    """
    Main function to convert LID-DS scenario directly to HDF5 format for ML training.
    
    Args:
        scenario_path: Path to LID-DS scenario directory
        output_dir: Output directory for HDF5 files and syscall table
        syscall_mappings: Optional pre-extracted syscall mappings. If None, will extract from scenario data.
    """
    logging.info("Starting LID-DS to HDF5 conversion...")
    logging.info(f"Source: {scenario_path}")
    logging.info(f"Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract syscall mappings if not provided
    if syscall_mappings is None:
        logging.info("No syscall mappings provided - extracting from scenario data...")
        syscall_mappings = extract_all_syscalls_from_scenario(scenario_path)
    
    # Create syscall table file
    syscall_table_path = os.path.join(output_dir, "syscall_64.tbl")
    create_syscall_table_from_mappings(syscall_mappings, syscall_table_path)
    
    # HDF5 output files
    normal_h5_path = os.path.join(output_dir, "0_normal.h5")
    
    # Remove existing HDF5 files to start over
    if os.path.exists(normal_h5_path):
        os.remove(normal_h5_path)
        logging.info(f"Removed existing file: {normal_h5_path}")
    
    # Process normal data (training files only)
    normal_count = process_scenario_data_by_type(
        scenario_path, "training", "training", syscall_mappings, normal_h5_path
    )
    
    logging.info("Conversion completed successfully!")
    logging.info(f"Created {normal_count} training sequences in: {normal_h5_path}")
    logging.info(f"Syscall table: {syscall_table_path}")
    
    # Print summary for trainer.py usage
    print("\n" + "="*60)
    print("LID-DS HDF5 CONVERSION COMPLETE")
    print("="*60)
    print("Files ready for ML training:")
    print(f"Training data: {normal_h5_path}")
    print(f"Syscall table: {syscall_table_path}")
    print("Update your trainer.py to use these paths:")
    print(f'H5LazyDataset("{normal_h5_path}", 0)  # Training data')
    print("Note: Only training data processed to avoid duplicates. Use separate test data for validation.")


def process_scenario_data_by_type(
    scenario_path: str,
    subdir: str,
    data_type: str,
    syscall_mappings: Dict[str, int],
    h5_output_path: str
) -> int:
    """
    Process LID-DS scenario data of a specific type and save to HDF5.
    Handles both directories and zip files efficiently.
    
    Args:
        scenario_path: Path to LID-DS scenario
        subdir: Subdirectory within scenario (e.g., "training", "test/normal_and_attack")
        data_type: Type of data ("training", "normal", "attack")
        syscall_mappings: Dictionary mapping syscall names to IDs
        h5_output_path: Path to output HDF5 file
        
    Returns:
        Number of sequences processed
    """
    data_path = os.path.join(scenario_path, subdir)
    sequence_count = 0
    
    if not os.path.exists(data_path):
        logging.warning(f"Data path does not exist: {data_path}")
        return sequence_count
    
    logging.info(f"Processing {data_type} data from: {data_path}")
    
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        
        if os.path.isdir(item_path):
            # Process directory - look for .sc file
            sc_file = os.path.join(item_path, f"{item}.sc")
            if os.path.exists(sc_file):
                # Extract syscalls from .sc file
                syscalls = extract_syscalls_from_sc_file(sc_file)
                if syscalls:
                    # Convert syscall names to IDs
                    syscall_ids = convert_syscalls_to_ids(syscalls, syscall_mappings)
                    
                    # Append to HDF5 file
                    append_seq_to_h5(syscall_ids, h5_output_path)
                    sequence_count += 1
                    
                    logging.info(f"Processed {data_type} sequence: {item} ({len(syscalls)} syscalls)")
        
        elif item.endswith(".zip"):
            # Process zip file temporarily
            zip_sequences = process_zip_temporarily(item_path, syscall_mappings, h5_output_path)
            sequence_count += zip_sequences
            logging.info(f"Processed zip file: {item} ({zip_sequences} sequences)")
    
    return sequence_count



def process_scenario_with_both_classes(
    scenario_path: str,
    output_dir: str,
    syscall_mappings: Optional[Dict[str, int]] = None
) -> None:
    """
    Process LID-DS scenario with both normal and attack data.
    
    Args:
        scenario_path: Path to LID-DS scenario directory
        output_dir: Output directory for HDF5 files and syscall table
        syscall_mappings: Optional pre-extracted syscall mappings
    """
    logging.info("Starting LID-DS to HDF5 conversion with both classes...")
    logging.info(f"Source: {scenario_path}")
    logging.info(f"Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract syscall mappings if not provided
    if syscall_mappings is None:
        logging.info("No syscall mappings provided - extracting from scenario data...")
        syscall_mappings = extract_all_syscalls_from_scenario(scenario_path)
    
    # Create syscall table file
    syscall_table_path = os.path.join(output_dir, "syscall_64.tbl")
    create_syscall_table_from_mappings(syscall_mappings, syscall_table_path)
    
    # HDF5 output files
    normal_h5_path = os.path.join(output_dir, "0_normal.h5")
    attack_h5_path = os.path.join(output_dir, "1_attack.h5")
    
    # Remove existing HDF5 files to start over
    for h5_path in [normal_h5_path, attack_h5_path]:
        if os.path.exists(h5_path):
            os.remove(h5_path)
            logging.info(f"Removed existing file: {h5_path}")
    
    # Process normal data (training files)
    normal_count = process_scenario_data_by_type(
        scenario_path, "training", "normal", syscall_mappings, normal_h5_path
    )
    
    # Process attack data (test/normal_and_attack files)
    attack_count = process_scenario_data_by_type(
        scenario_path, "test/normal_and_attack", "attack", syscall_mappings, attack_h5_path
    )
    
    logging.info("Conversion completed successfully!")
    logging.info(f"Created {normal_count} normal sequences in: {normal_h5_path}")
    logging.info(f"Created {attack_count} attack sequences in: {attack_h5_path}")
    logging.info(f"Syscall table: {syscall_table_path}")
    
    # Print summary for trainer.py usage
    print("\n" + "="*60)
    print("LID-DS HDF5 CONVERSION COMPLETE")
    print("="*60)
    print("Files ready for ML training:")
    print(f"Normal data: {normal_h5_path} ({normal_count} sequences)")
    print(f"Attack data: {attack_h5_path} ({attack_count} sequences)")
    print(f"Syscall table: {syscall_table_path}")
    print("Update your trainer.py to use these paths:")
    print(f'normal_dataset = H5LazyDataset("{normal_h5_path}", 0)')
    print(f'attack_dataset = H5LazyDataset("{attack_h5_path}", 1)')


# Main function to run the script
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Fixed paths, adjust if necessary
    scenarios_base = os.path.join(os.path.dirname(__file__), "SCENARIOS")
    output_dir = os.path.join(os.path.dirname(__file__), "processed_lid_data")
    
    # Max. syscalls collection
    print("=" * 60)
    print("LID-DS LOADER - MAXIMUM SYSCALLS COLLECTION MODE")
    print("=" * 60)
    print("Processing all available scenarios to gather maximum syscalls...")
    print(f"Scenarios directory: {scenarios_base}")
    print(f"Output directory: {output_dir}")
    
    # Check if scenarios directory exists
    assert os.path.exists(scenarios_base), f"Scenarios directory does not exist: {scenarios_base}"
    
    # Discover all scenarios
    scenarios = discover_all_scenarios(scenarios_base)
    
    assert scenarios, "No scenarios found in SCENARIOS directory."
    
    # Process all scenarios
    print(f"Found {len(scenarios)} scenarios:")
    for scenario in scenarios:
        print(f"  - {os.path.basename(scenario)}")
    
    print("\nProcessing all scenarios to maximize syscalls collection...")
    
    # Process all scenarios to gather maximum syscalls
    process_all_scenarios_to_hdf5(output_dir, scenarios_base)
    
    print("\n" + "=" * 60)
    print("SUCCESS: LID-DS SYSCALLS COLLECTION COMPLETE")
    print("=" * 60)
    print(f"All {len(scenarios)} scenarios processed successfully!")
    print(f"Maximum syscalls dataset available in: {output_dir}")
    print("\nFiles ready for ML training:")
    print("  - 0_normal.h5 (normal syscall sequences)")
    print("  - 1_attack.h5 (attack syscall sequences)")
    print("  - syscall_64.tbl (syscall vocabulary)")
    print("\nTrainer.py can now use the complete syscalls dataset!")
