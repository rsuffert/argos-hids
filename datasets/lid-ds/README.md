# LID-DS Loader

A pipeline for converting LID-DS syscall datasets into ML-ready HDF5 format. This loader processes syscall traces from LID-DS scenarios.

## Features

- **Automatic Processing**: Discovers and processes all LID-DS scenarios automatically
- **Zip Support**: Efficiently handles compressed scenario data with temporary extraction
- **Disk Efficient**: Processes zip files temporarily to minimize disk usage
- **Batch Processing**: Process single scenarios or all scenarios at once

## Installation

```bash
cd datasets/lid-ds
pip install -r requirements.txt
```

## Usage

### Process All Scenarios
```bash
# Auto-discover and process all scenarios in SCENARIOS/ directory
python3 lid_loader.py --all

# Process all scenarios with custom output directory
python3 lid_loader.py --all /path/to/output
```

### Process Single Scenario
```bash
# Process specific scenario
python3 lid_loader.py /path/to/scenario

# Process with custom output directory
python3 lid_loader.py /path/to/scenario /path/to/output
```

### Auto-Detection
```bash
# Auto-detect scenarios and process accordingly
python3 lid_loader.py
```
### 