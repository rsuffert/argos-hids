# LID-DS Dataset Loader

This module provides a loader for the Leipzig Intrusion Detection System (LID-DS) dataset, converting syscall traces from any LID-DS scenario into a format compatible with the machine learning pipeline.

## Features

- **Dynamic syscall extraction**: Automatically discovers and maps syscalls from any LID-DS scenario
- **Generic compatibility**: Works with any scenario (CVE-2014-0160, CVE-2017-7529, etc.)
- **ML-ready output data**: Generates raw sequences, syscall tables, Excel baselines, and HDF5 files

## Usage

```bash
# Convert any LID-DS scenario to ML pipeline format
python3 datasets/lid-ds/loader.py SCENARIOS/CVE-2014-0160 output_directory
```

## Requirements

```bash
cd datasets/lid-ds
pip install -r requirements.txt
```
