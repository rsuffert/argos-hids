# LID-DS Loader

A pipeline for converting LID-DS syscall scenario ZIP files into ML-ready HDF5 files compatible with ARGOS.

## Requirements
Install project dependencies:
```bash
poetry install
```

## Run
From the dataparse/lid directory run the loader with poetry:
```bash
cd dataparse/lid
poetry run python3 loader.py
```

## Output
Generated files in `LID_DATA_DIR`:
- `syscall_dict.pkl` — cached syscall with id mapping  
- `0_training.h5`, `1_training.h5`  
- `0_validation.h5`, `1_validation.h5`  
- `0_test.h5`, `1_test.h5`
- ``mapping.csv` — syscall name to ID mapping used

HDF5 format matches DongTing loader for interoperability across ARGOS.