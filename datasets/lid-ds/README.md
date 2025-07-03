# LID-DS Dataset Loader

This module provides a loader for the Leipzig Intrusion Detection System (LID-DS) dataset, converting syscall traces from any LID-DS scenario into a format compatible with the machine learning pipeline.

## Features

- **Dynamic syscall extraction**: Automatically discovers and maps syscalls from any LID-DS scenario
- **Generic compatibility**: Works with any scenario (CVE-2014-0160, CVE-2017-7529, etc.)

## Usage

```bash
# Convert LID-DS scenario to the pipeline format
python3 lid_loader.py
```
```bash
# Convert using the Dongting loader to ML pipeline format 
python3 loader.py
```

## Requirements

```bash
cd datasets/lid-ds
pip install -r requirements.txt
```
