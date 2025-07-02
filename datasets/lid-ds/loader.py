#!/usr/bin/env python3
"""
LID-DS to Pipeline Adapter
Converts LID-DS syscall data to format compatible with ML pipeline loader.
Dynamically extracts syscalls from any LID-DS scenario.
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Tuple
import logging

def create_syscall_table_from_mappings(syscall_mappings: Dict[str, int], output_path: str) -> None:
    """
    Create a syscall table file compatible with the loader from LID-DS syscall mappings
    
    Args:
        syscall_mappings: Dictionary mapping syscall names to IDs
        output_path: Path where to save the syscall table file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Syscall table generated from LID-DS scenario data\n")
        f.write("# Format: ID ABI NAME\n")
        f.write("#\n")
        
        # Add UNKNOWN syscall at ID 0
        f.write("0\tcommon\tUNKNOWN\n")
        
        # Add all extracted syscalls
        for syscall_name, syscall_id in sorted(syscall_mappings.items(), key=lambda x: x[1]):
            f.write(f"{syscall_id}\tcommon\t{syscall_name}\n")
    
    logging.info(f"Created syscall table with {len(syscall_mappings)} syscalls at: {output_path}")


def extract_syscalls_from_sc_file(sc_file_path: str) -> List[str]:
    """
    Extract syscalls from a .sc file (LID-DS format)
    
    Args:
        sc_file_path: Path to the .sc file
        
    Returns:
        List of syscall names in order
    """
    syscalls = []
    
    try:
        with open(sc_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    syscall_name = parts[5]  # syscall name is at index 5
                    syscalls.append(syscall_name)
    except Exception as e:
        logging.error(f"Error reading {sc_file_path}: {e}")
    
    return syscalls


def create_raw_sequence_files(scenario_path: str, output_dir: str, separator: str = "|") -> List[str]:
    """
    Convert LID-DS .sc files to raw sequence files compatible with the loader
    
    Args:
        scenario_path: Path to the LID-DS scenario directory
        output_dir: Directory where to save the raw sequence files
        separator: Separator to use between syscalls (default "|")
        
    Returns:
        List of created file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    
    # Process training files (normal data)
    normal_files = create_raw_sequence_files_by_type(
        scenario_path, output_dir, "training", "normal", separator
    )
    created_files.extend(normal_files)
    
    # Process test files (attack data)
    attack_files = create_raw_sequence_files_by_type(
        scenario_path, output_dir, "test/normal_and_attack", "attack", separator
    )
    created_files.extend(attack_files)
    
    return created_files


def create_baseline_excel(sequence_files: List[str], output_path: str) -> None:
    """
    Create a baseline Excel file with labels and classes for the sequence files
    
    Args:
        sequence_files: List of created sequence file paths
        output_path: Path where to save the baseline Excel file
    """
    data = []
    
    for file_path in sequence_files:
        filename = os.path.basename(file_path)
        
        # Extract bug name (remove .log extension)
        bug_name = filename[:-4] if filename.endswith('.log') else filename
        
        # Determine label and class based on filename
        if filename.startswith('normal_'):
            label = 0  # Normal = 0
            class_name = "normal"
        elif filename.startswith('attack_'):
            label = 1  # Attack = 1  
            class_name = "attack"
        else:
            label = 0  # Default to normal
            class_name = "normal"
        
        data.append({
            'kcb_bug_name': bug_name,
            'kcb_seq_lables': label,
            'kcb_seq_class': class_name
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    
    logging.info(f"Created baseline Excel file with {len(data)} entries at: {output_path}")
    logging.info(f"Normal samples: {len([d for d in data if d['kcb_seq_lables'] == 0])}")
    logging.info(f"Attack samples: {len([d for d in data if d['kcb_seq_lables'] == 1])}")


def setup_pipeline_directories(base_output_dir: str) -> Tuple[str, str, str, str]:
    """
    Setup directory structure compatible with the loader pipeline
    
    Args:
        base_output_dir: Base directory for all pipeline files
        
    Returns:
        Tuple of (normal_dir, abnormal_dir, syscall_table_path, baseline_path)
    """
    os.makedirs(base_output_dir, exist_ok=True)
    
    normal_dir = os.path.join(base_output_dir, "Normal_data")
    abnormal_dir = os.path.join(base_output_dir, "Abnormal_data") 
    syscall_table_path = os.path.join(base_output_dir, "syscall_64.tbl")
    baseline_path = os.path.join(base_output_dir, "Baseline.xlsx")
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)
    
    return normal_dir, abnormal_dir, syscall_table_path, baseline_path


def extract_all_syscalls_from_scenario(scenario_path: str) -> Dict[str, int]:
    """
    Extract all unique syscalls from a LID-DS scenario and create mappings
    
    Args:
        scenario_path: Path to LID-DS scenario directory
        
    Returns:
        Dictionary mapping syscall names to IDs (starting from 1, 0 reserved for UNKNOWN)
    """
    logging.info(f"Extracting syscalls from scenario: {scenario_path}")
    
    all_syscalls = set()
    file_count = 0
    
    # Search for all .sc files in the scenario
    for root, dirs, files in os.walk(scenario_path):
        for file in files:
            if file.endswith('.sc'):
                sc_file_path = os.path.join(root, file)
                file_count += 1
                logging.info(f"Processing {file_count}: {file}")
                
                syscalls = extract_syscalls_from_sc_file(sc_file_path)
                all_syscalls.update(syscalls)
    
    # Create mappings (alphabetically sorted for consistency)
    unique_syscalls = sorted(list(all_syscalls))
    syscall_mappings = {}
    
    for i, syscall_name in enumerate(unique_syscalls, 1):
        syscall_mappings[syscall_name] = i
    
    logging.info(f"Found {len(unique_syscalls)} unique syscalls across {file_count} files")
    logging.info(f"Syscalls: {', '.join(unique_syscalls[:10])}{'...' if len(unique_syscalls) > 10 else ''}")
    
    return syscall_mappings


def convert_lid_ds_to_pipeline_format(
    scenario_path: str,
    output_dir: str,
    syscall_mappings=None
) -> None:
    """
    Main function to convert LID-DS data to pipeline-compatible format
    
    Args:
        scenario_path: Path to LID-DS scenario directory (e.g., SCENARIOS/CVE-2014-0160)
        output_dir: Output directory for pipeline-compatible files
        syscall_mappings: Optional pre-extracted syscall mappings. If None, will extract from scenario data.
    """
    logging.info("Starting LID-DS to Pipeline conversion...")
    logging.info(f"Source: {scenario_path}")
    logging.info(f"Output: {output_dir}")
    
    # Setup directory structure
    normal_dir, abnormal_dir, syscall_table_path, baseline_path = setup_pipeline_directories(output_dir)
    
    # Extract syscall mappings if not provided
    if syscall_mappings is None:
        logging.info("No syscall mappings provided - extracting from scenario data...")
        syscall_mappings = extract_all_syscalls_from_scenario(scenario_path)
    
    # Create syscall table file
    create_syscall_table_from_mappings(syscall_mappings, syscall_table_path)
    
    # Convert .sc files to raw sequence files
    logging.info("Converting .sc files to raw sequence format...")
    all_files = []
    
    # Process normal data (training files)
    normal_files = create_raw_sequence_files_by_type(
        scenario_path, normal_dir, "training", "normal", "|"
    )
    all_files.extend(normal_files)
    
    # Process attack data (test files)  
    attack_files = create_raw_sequence_files_by_type(
        scenario_path, abnormal_dir, "test/normal_and_attack", "attack", "|"
    )
    all_files.extend(attack_files)
    
    # Create baseline Excel file
    logging.info("Creating baseline Excel file...")
    create_baseline_excel(all_files, baseline_path)
    
    logging.info("Conversion completed successfully!")
    logging.info(f"Created {len(normal_files)} normal sequence files")
    logging.info(f"Created {len(attack_files)} attack sequence files")
    logging.info(f"Pipeline files saved in: {output_dir}")
    
    # Print environment variables for the loader
    print("\n" + "="*60)
    print("PIPELINE SETUP COMPLETE")
    print("="*60)
    print("Set these environment variables before running the loader:")
    print(f'export SYSCALL_TBL_PATH="{syscall_table_path}"')
    print(f'export NORMAL_DATA_FOLDER_PATH="{normal_dir}"') 
    print(f'export ABNORMAL_DATA_FOLDER_PATH="{abnormal_dir}"')
    print(f'export BASELINE_XLSX_PATH="{baseline_path}"')
    print("\nOr copy your loader.py to the output directory and run it from there.")


def create_raw_sequence_files_by_type(
    scenario_path: str, 
    output_dir: str, 
    subdir: str, 
    file_prefix: str, 
    separator: str
) -> List[str]:
    """
    Helper function to create raw sequence files for a specific data type
    
    Args:
        scenario_path: Path to LID-DS scenario
        output_dir: Output directory for files
        subdir: Subdirectory within scenario (e.g., "training", "test/normal_and_attack")
        file_prefix: Prefix for output files (e.g., "normal", "attack")
        separator: Separator between syscalls
        
    Returns:
        List of created file paths
    """
    created_files: List[str] = []
    data_path = os.path.join(scenario_path, subdir)
    
    if not os.path.exists(data_path):
        logging.warning(f"Data path does not exist: {data_path}")
        return created_files
    
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            sc_file = os.path.join(item_path, f"{item}.sc")
            if os.path.exists(sc_file):
                syscalls = extract_syscalls_from_sc_file(sc_file)
                if syscalls:
                    output_file = os.path.join(output_dir, f"{file_prefix}_{item}.log")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(separator.join(syscalls))
                    created_files.append(output_file)
                    logging.info(f"Created {file_prefix} file: {output_file} ({len(syscalls)} syscalls)")
    
    return created_files


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Default paths - adjust as needed
    scenario_path = os.path.join(os.path.dirname(__file__), "..", "..", "SCENARIOS", "CVE-2014-0160")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "pipeline_data")
    
    if len(sys.argv) > 1:
        scenario_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    if not os.path.exists(scenario_path):
        print(f"Error: Scenario path does not exist: {scenario_path}")
        print("Usage: python loader.py [scenario_path] [output_dir]")
        sys.exit(1)
    
    try:
        # Dynamic syscall extraction
        convert_lid_ds_to_pipeline_format(scenario_path, output_dir)
        print("\nSuccessfully converted LID-DS data to pipeline format.")
        print(f"Pipeline files available in: {output_dir}")
        
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
