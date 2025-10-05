# LID-DS Loader

A pipeline for converting LID-DS syscall datasets into GNN format. This loader processes syscall traces from LID-DS scenarios. All scenarios from the LID dataset are available [here](https://drive.proton.me/urls/BWKRGQK994#fCK9JKL93Sjm).

## Usage

```bash
# H5 dataset with arguments
python loader.py -n normal.h5 -a attack.h5 -p

# H5 dataset with environment variables
export LID_NORMAL="/path/to/normal.h5"
export LID_ATTACK="/path/to/attack.h5"
python loader.py -p

# LID-DS dataset and preprocessing
python loader.py -l datasets/folder -p

# Extract only (no graph preprocessing)
python loader.py -n normal.h5 -a attack.h5
python loader.py -l datasets/folder
```

## Args

- `-n, --normal` - Path to H5 file with normal traces
- `-a, --attack` - Path to H5 file with attack traces  
- `-l, --lidds` - Path to LID-DS dataset directory
- `-p, --preprocess` - Preprocess traces to graphs (.pkl dictionary files)