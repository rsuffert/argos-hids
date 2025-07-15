# Syscall GAN Generator

## Objective
Generate synthetic syscall sequences to test the generalization capability of the Argos-Hids detection model. The GAN learns patterns from real DongTing dataset and produces realistic synthetic data for model testing.

## Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```
**Generate synthetic data:**
```bash
python generator.py
```

## Output Files:
- `Synthetic_Normal_DTDS-test.h5` 
- `Synthetic_Attach_DTDS-test.h5` 


