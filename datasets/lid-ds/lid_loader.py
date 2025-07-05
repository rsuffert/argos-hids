#!/usr/bin/env python3
"""LID-DS Loader with External Syscall Table and Exploit-based Labeling."""

import logging
import h5py
import numpy as np
import zipfile
import tempfile
import shutil
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import the syscall table parser from dongting loader
sys.path.append(str(Path(__file__).parent.parent / "dongting"))
from loader import parse_syscall_tbl 

class LIDSLoader:
    """LID-DS dataset loader with external syscall mapping and exploit-based labeling."""
    
    def __init__(
        self,
        external_syscall_table: str,
        scenarios_base: str = "SCENARIOS",
        output_dir: str = "processed_lid_data"
    ) -> None:
        """
        Initialize the LIDSLoader with paths to the external syscall table, scenarios base directory,
        and output directory.

        Args:
            external_syscall_table (str): Path to the external syscall table file.
            scenarios_base (str, optional): Path to the scenarios base directory. Defaults to "SCENARIOS".
            output_dir (str, optional): Path to the output directory. Defaults to "processed_lid_data".
        """
        self.external_syscall_table = Path(external_syscall_table)
        self.scenarios_base = Path(scenarios_base)
        self.output_dir = Path(output_dir)
        self.syscall_mappings = self._load_external_syscall_table()
        self.missing_syscalls: set[str] = set()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_external_syscall_table(self) -> Dict[str, int]:
        """Load syscall mappings from external syscall table using dongting parser."""
        if not self.external_syscall_table.exists():
            raise FileNotFoundError(f"External syscall table not found: {self.external_syscall_table}")
        
        # Use the existing parse_syscall_tbl function from dongting loader
        syscall_mappings = parse_syscall_tbl(str(self.external_syscall_table))
        
        logging.info(f"Loaded {len(syscall_mappings)} syscall mappings from {self.external_syscall_table}")
        return syscall_mappings
    
    def _extract_syscalls_from_sc_file(self, sc_file_path: Path) -> List[str]:
        """Extract syscall names from .sc trace file."""
        try:
            with open(sc_file_path, "r") as f:
                return self._parse_sc_lines(f, sc_file_path)
        except Exception as e:
            logging.error(f"Error reading .sc file {sc_file_path}: {e}")
            return []

    from typing import TextIO

    def _parse_sc_lines(self, file_handle: TextIO, sc_file_path: Path) -> List[str]:
        """Parse lines from SC file and extract syscalls."""
        syscalls = []
        
        for line_num, line in enumerate(file_handle, 1):
            syscall_name = self._extract_syscall_from_line(line, line_num, sc_file_path)
            if syscall_name:
                syscalls.append(syscall_name)
        
        return syscalls

    def _extract_syscall_from_line(self, line: str, line_num: int, sc_file_path: Path) -> str:
        """Extract syscall name from a single line."""
        line = line.strip()
        if not line:
            return ""
        
        try:
            parts = line.split()
            if len(parts) >= 6:
                return parts[5].strip()
        except (IndexError, ValueError) as e:
            logging.debug(f"Error parsing line {line_num} in {sc_file_path}: {e}")
        
        return ""
    
    def _convert_syscalls_to_ids(self, syscalls: List[str]) -> List[int]:
        """Convert syscall names to their integer IDs using external mapping."""
        syscall_ids = []
        for syscall_name in syscalls:
            if syscall_name in self.syscall_mappings:
                syscall_ids.append(self.syscall_mappings[syscall_name])
            else:
                # Track missing syscalls
                self.missing_syscalls.add(syscall_name)
                syscall_ids.append(0)  # Unknown syscall
                logging.warning(f"Unknown syscall not in external table: {syscall_name}")
        
        return syscall_ids
    
    def _append_sequence_to_h5(self, sequence: List[int], h5_path: Path) -> None:
        """Append a sequence to HDF5 file."""
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
    
    def _extract_label_from_json(self, json_path: Path) -> str:
        """Extract label from JSON metadata based on exploit field."""
        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
            
            return self._parse_exploit_field(metadata.get("exploit"), json_path)
            
        except Exception as e:
            logging.error(f"Error parsing JSON {json_path}: {e}")
            return "error"

    def _parse_exploit_field(self, exploit_value: Optional[object], json_path: Path) -> str:
        """Parse the exploit field value."""
        if exploit_value is None:
            logging.warning(f"No valid 'exploit' field found in {json_path}")
            return "unknown"
        
        if isinstance(exploit_value, bool):
            return "attack" if exploit_value else "normal"
        
        if isinstance(exploit_value, str):
            exploit_str = exploit_value.lower()
            if exploit_str in ["true", "1", "yes"]:
                return "attack"
            if exploit_str in ["false", "0", "no"]:
                return "normal"
        
        logging.warning(f"Invalid exploit value '{exploit_value}' in {json_path}")
        return "unknown"
    
    def _process_zip_file(self, zip_path: Path) -> Tuple[str, List[int]]:
        """Process a single ZIP file and return label and syscall sequence."""
        base_name = zip_path.stem
        temp_dir = None
        
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"lid_ds_{base_name}_"))
            
            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(temp_dir)
            
            # Extract label from JSON
            json_path = temp_dir / f"{base_name}.json"
            label = self._extract_label_from_json(json_path) if json_path.exists() else "unknown"
            
            # Extract syscalls from .sc file
            sc_path = temp_dir / f"{base_name}.sc"
            syscalls = []
            if sc_path.exists():
                syscalls = self._extract_syscalls_from_sc_file(sc_path)
            
            # Convert to IDs
            syscall_ids = self._convert_syscalls_to_ids(syscalls) if syscalls else []
            
            return label, syscall_ids
            
        except Exception as e:
            logging.error(f"Error processing {zip_path}: {e}")
            return "error", []
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _process_folder(self, folder_path: Path, folder_type: str) -> Dict[str, int]:
        """Process all ZIP files in a folder."""
        if not folder_path.exists():
            logging.warning(f"Folder does not exist: {folder_path}")
            return {}
        
        zip_files = self._get_valid_zip_files(folder_path)
        if not zip_files:
            logging.warning(f"No ZIP files found in {folder_path}")
            return {}
        
        return self._process_zip_files(zip_files, folder_type)

    def _get_valid_zip_files(self, folder_path: Path) -> List[Path]:
        """Get valid ZIP files excluding MACOS artifacts."""
        return [
            f for f in folder_path.glob("*.zip") 
            if not any(part.startswith(("__MACOSX", "._")) for part in f.parts)
        ]

    def _process_zip_files(self, zip_files: List[Path], folder_type: str) -> Dict[str, int]:
        """Process a list of ZIP files and return counts."""
        counts = {"normal": 0, "attack": 0, "unknown": 0, "error": 0}
        
        logging.info(f"Processing {len(zip_files)} ZIP files from {folder_type} folder")
        
        for i, zip_file in enumerate(zip_files):
            label, syscall_ids = self._process_zip_file(zip_file)
            
            if self._should_save_sequence(syscall_ids, label):
                output_file = self._get_output_file(folder_type, label)
                self._append_sequence_to_h5(syscall_ids, output_file)
                counts[label] += 1
            else:
                counts[label] += 1
                if label not in ["normal", "attack"]:
                    logging.debug(f"Skipped {zip_file.name}: label={label}, syscalls={len(syscall_ids)}")
            
            if (i + 1) % 50 == 0:
                logging.info(f"  Processed {i + 1}/{len(zip_files)} files")
        
        return counts

    def _should_save_sequence(self, syscall_ids: List[int], label: str) -> bool:
        """Check if sequence should be saved."""
        return bool(syscall_ids) and label in ["normal", "attack"]

    def _get_output_file(self, folder_type: str, label: str) -> Path:
        """Get output file path based on folder type and label."""
        if folder_type in ["test", "training"]:
            return self.output_dir / f"train_{label}.h5"
        # validation
        return self.output_dir / f"val_{label}.h5"
    
    def process_scenario(self, scenario_path: Path) -> Dict[str, Dict[str, int]]:
        """Process a single scenario (all folders: test, training, validation)."""
        scenario_name = scenario_path.name
        logging.info(f"Processing scenario: {scenario_name}")
        
        results = {}
        
        # Process each folder type
        for folder_name in ["test", "training", "validation"]:
            folder_path = scenario_path / folder_name
            if folder_path.exists():
                # For test folder, also check subfolders
                if folder_name == "test":
                    # Process test/normal and test/normal_and_attack separately
                    normal_folder = folder_path / "normal"
                    attack_folder = folder_path / "normal_and_attack"
                    
                    if normal_folder.exists():
                        results["test_normal"] = self._process_folder(normal_folder, "test")
                    if attack_folder.exists():
                        results["test_attack"] = self._process_folder(attack_folder, "test")
                else:
                    results[folder_name] = self._process_folder(folder_path, folder_name)
        
        return results
    
    def process_all_scenarios(self) -> None:
        """Process all scenarios in the scenarios base directory."""
        scenarios = self._get_valid_scenarios()
        if not scenarios:
            logging.error(f"No scenarios found in {self.scenarios_base}")
            return

        logging.info(f"Found {len(scenarios)} scenarios: {[s.name for s in scenarios]}")
        
        self._clear_output_files()
        total_results = self._initialize_results()
        
        for scenario_path in scenarios:
            scenario_results = self.process_scenario(scenario_path)
            self._aggregate_results(scenario_results, total_results)
            logging.info(f"Scenario {scenario_path.name} completed")

        self._report_missing_syscalls()
        self._print_summary(total_results)

    def _get_valid_scenarios(self) -> List[Path]:
        """Get valid scenario directories."""
        return [
            p for p in self.scenarios_base.iterdir() 
            if p.is_dir() and not p.name.startswith(("__MACOSX", "."))
        ]

    def _initialize_results(self) -> Dict[str, int]:
        """Initialize results dictionary."""
        return {
            "train_normal": 0, "train_attack": 0,
            "val_normal": 0, "val_attack": 0
        }
    
    def _clear_output_files(self) -> None:
        """Remove existing output files before processing."""
        output_files = [
            self.output_dir / "train_normal.h5",
            self.output_dir / "train_attack.h5",
            self.output_dir / "val_normal.h5",
            self.output_dir / "val_attack.h5"
        ]
        for output_file in output_files:
            if output_file.exists():
                output_file.unlink()

    def _aggregate_results(self, scenario_results: Dict[str, Dict[str, int]], total_results: Dict[str, int]) -> None:
        """Aggregate results from a scenario into the total results."""
        for folder_type, counts in scenario_results.items():
            for label, count in counts.items():
                if label in ["normal", "attack"]:
                    if folder_type in ["test_normal", "test_attack", "training"]:
                        key = f"train_{label}"
                    else:  # validation
                        key = f"val_{label}"
                    total_results[key] += count

    def _report_missing_syscalls(self) -> None:
        """Report and save missing syscalls if any."""
        if self.missing_syscalls:
            logging.warning(f"Found {len(self.missing_syscalls)} syscalls not in external table:")
            for syscall in sorted(self.missing_syscalls):
                logging.warning(f"  Missing: {syscall}")

            missing_file = self.output_dir / "missing_syscalls.txt"
            with open(missing_file, "w") as f:
                f.write("# Syscalls found in LID-DS but missing from external table\n")
                for syscall in sorted(self.missing_syscalls):
                    f.write(f"{syscall}\n")
            logging.info(f"Missing syscalls saved to: {missing_file}")
    
    def _print_summary(self, results: Dict[str, int]) -> None:
        """Print processing summary."""
        print("\n" + "="*60)
        print("LID-DS PROCESSING COMPLETE")
        print("="*60)
        print("DATASETS CREATED:")
        print(f"Output directory: {self.output_dir}")
        print(f"External syscall table: {self.external_syscall_table}")
        print("\nTRAINING DATA:")
        print(f"  train_normal.h5: {results['train_normal']} sequences")
        print(f"  train_attack.h5: {results['train_attack']} sequences")
        print("\nVALIDATION DATA:")
        print(f"  val_normal.h5: {results['val_normal']} sequences")
        print(f"  val_attack.h5: {results['val_attack']} sequences")
        
        total_train = results["train_normal"] + results["train_attack"]
        total_val = results["val_normal"] + results["val_attack"]
        
        if results["train_attack"] > 0:
            train_ratio = results["train_normal"] / results["train_attack"]
            print(f"\nTRAINING BALANCE: {train_ratio:.1f}:1 (normal:attack)")
        
        if results["val_attack"] > 0:
            val_ratio = results["val_normal"] / results["val_attack"]
            print(f"VALIDATION BALANCE: {val_ratio:.1f}:1 (normal:attack)")
        
        print(f"\nTOTAL: {total_train} training + {total_val} validation sequences")
        
        if self.missing_syscalls:
            print(f"\nATTENTION: {len(self.missing_syscalls)} syscalls missing from external table")
            print("   Check missing_syscalls.txt for details")

def main() -> None:
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Configs
    external_syscall_table = Path("../syscall_64.tbl")  # Adjust path to your external table
    scenarios_base = Path("SCENARIOS")
    output_dir = Path("processed_lid_data")
    
    # Validate inputs
    if not external_syscall_table.exists():
        logging.error(f"External syscall table not found: {external_syscall_table}")
        logging.error("Please provide path to existing syscall_64.tbl file")
        return
    
    if not scenarios_base.exists():
        logging.error(f"Scenarios directory not found: {scenarios_base}")
        return
    
    # Create loader and process
    loader = LIDSLoader(
        external_syscall_table=str(external_syscall_table),
        scenarios_base=str(scenarios_base),
        output_dir=str(output_dir)
    )
    
    loader.process_all_scenarios()
    
    print("\nProcessing complete! Ready for ML training:")
    print("   - train_normal.h5 and train_attack.h5 (training)")
    print("   - val_normal.h5 and val_attack.h5 (validation)")

if __name__ == "__main__":
    main()