# LID-DS Loader

A pipeline for converting LID-DS syscall datasets into ML-ready HDF5 format. This loader processes syscall traces from LID-DS scenarios.

## Requirements 

```bash
cd datasets/lid-ds
pip install -r requirements.txt
```
## Usage

### Process Scenarios
```bash
# Process all scenarios in SCENARIOS/ directory
python3 lid_loader.py 
```