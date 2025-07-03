"""
Module for training an LSTM intrusion detection model on DongTing and LID-DS datasets using
PyTorch Lightning.
"""

import os
from typing import Tuple, List
from dataclasses import dataclass
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchmetrics import F1Score, Accuracy
import h5py
import numpy as np
import random

torch.backends.cudnn.enabled = False

# =============================
# Hyperparameters and constants
# =============================

INPUT_SIZE = 1              # Each element in the sequence is a scalar
HIDDEN_SIZE = 64            # Number of hidden units in the LSTM
NUM_LAYERS = 2              # Number of stacked LSTM layers
NUM_CLASSES = 2             # Output classes: normal or attack
LEARNING_RATE = 1e-3        # Learning rate for the optimizer
BATCH_SIZE = 64             # Batch size for DataLoader
MAX_EPOCHS = 10             # Number of training epochs
TRAIN_ATTACK_SPLIT = 0.6    # Proportion of attack data used for training
MAX_SEQ_LEN = 512           # Maximum sequence length for padding/truncation

# ====================
# Collate function
# ====================

def collate(batch: List[Tuple[np.ndarray, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    """Custom collate function to pad sequences and prepare batches."""
    sequences, labels = map(list, zip(*batch))
    # PyTorch expects tensors to be floating-point, even though they are scalars in our case
    sequences = [torch.as_tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32)
                 for seq in sequences]
    lengths = torch.as_tensor(
        [min(len(seq), MAX_SEQ_LEN)
         for seq in sequences],
        dtype=torch.long
    )
    # padding sequences with zeros to the maximum length in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels_ = torch.as_tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels_

# ====================
# Model definition
# ====================

@dataclass
class LSTMConfig:
    """Configuration for the LSTM classifier."""
    input_size: int
    hidden_size: int
    num_layers: int
    num_classes: int
    lr: float

class LSTMClassifier(pl.LightningModule):
    """LSTM-based classifier using PyTorch Lightning."""

    def __init__(self, config: LSTMConfig) -> None:
        """Initialize the LSTM classifier with the given configuration."""
        super().__init__()
        self.save_hyperparameters(config.__dict__)
        self.lr = config.lr

        self.lstm = torch.nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)
        class_weights = torch.tensor([1.0, 2.0])  # Weight attack class 2x more
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass through LSTM and classification layer."""
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed_input)
        return self.fc(h_n[-1])

    def shared_step(self, batch: Tuple[Tensor, Tensor, Tensor], step_type: str) -> Tensor:
        """Shared logic for training and validation steps."""
        sequences, lengths, labels = batch
        sequences = sequences.unsqueeze(-1)
        outputs = self(sequences, lengths)
        preds = outputs.argmax(dim=1)

        loss = self.criterion(outputs, labels)
        acc = self.accuracy(preds, labels)
        f1 = self.f1(preds, labels)
        self.log(f"{step_type}_loss", loss, prog_bar=step_type == "val")
        self.log(f"{step_type}_acc", acc, prog_bar=step_type == "val")
        self.log(f"{step_type}_f1", f1, prog_bar=step_type == "val")

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], _batch_idx: int) -> Tensor:
        """Training step for one batch."""
        return self.shared_step(batch, step_type="train")

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], _batch_idx: int) -> None:
        """Validation step for one batch."""
        self.shared_step(batch, step_type="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ====================
# Data loading
# ====================

# DongTing dataset paths
DONGTING_BASE_DIR = os.path.join("..", "..", "datasets", "dongting")

# LID-DS dataset paths  
LID_DATA_DIR = os.path.join("..", "..", "datasets", "lid-ds", "processed_lid_data")

# Check which datasets are available
DONGTING_AVAILABLE = os.path.exists(DONGTING_BASE_DIR) and os.path.isdir(DONGTING_BASE_DIR)
LID_DS_AVAILABLE = os.path.exists(LID_DATA_DIR) and os.path.isdir(LID_DATA_DIR)

DONGTING_AVAILABLE = False  # Temporarily set to False for testing

print(f"DongTing dataset available: {DONGTING_AVAILABLE}")
print(f"LID-DS dataset available: {LID_DS_AVAILABLE}")

class H5LazyDataset(torch.utils.data.Dataset):
    """Lazy dataset for reading sequences from an HDF5 file."""

    def __init__(self, h5_path: str, label: int) -> None:
        """
        Initializes the object with the given HDF5 file path and label.

        Args:
            h5_path (str): Path to the HDF5 file containing the sequences.
            label (int): Label associated with the data.

        Raises:
            AssertionError: If the specified HDF5 file does not exist.

        Attributes:
            h5_path (str): Path to the HDF5 file.
            length (int): Number of sequences in the HDF5 file.
            label (int): Label associated with the data.
        """
        assert os.path.exists(h5_path), f"HDF5 file not found at '{h5_path}'"
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as h5f:
            self.length = len(h5f["sequences"])
        self.label = label

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        with h5py.File(self.h5_path, "r") as h5f:
            sequence = h5f["sequences"][idx]
        return sequence, self.label

# Build datasets dynamically based on availability
train_datasets = []
valid_datasets = []


# dongting_normal_train = os.path.join(DONGTING_BASE_DIR, "Normal_DTDS-train.h5")
# dongting_attack_train = os.path.join(DONGTING_BASE_DIR, "Attach_DTDS-train.h5")
# dongting_normal_valid = os.path.join(DONGTING_BASE_DIR, "Normal_DTDS-validation.h5")
# dongting_attack_valid = os.path.join(DONGTING_BASE_DIR, "Attach_DTDS-validation.h5")


# assert DONGTING_AVAILABLE, f"DongTing dataset directory not found at {DONGTING_BASE_DIR}"
# assert os.path.exists(dongting_normal_train), f"DongTing normal train data not found at {dongting_normal_train}"
# assert os.path.exists(dongting_attack_train), f"DongTing attack train data not found at {dongting_attack_train}"
# assert os.path.exists(dongting_normal_valid), f"DongTing normal validation data not found at {dongting_normal_valid}"
# assert os.path.exists(dongting_attack_valid), f"DongTing attack validation data not found at {dongting_attack_valid}"


# train_datasets.append(H5LazyDataset(dongting_normal_train, 0))
# train_datasets.append(H5LazyDataset(dongting_attack_train, 1))
# valid_datasets.append(H5LazyDataset(dongting_normal_valid, 0))
# valid_datasets.append(H5LazyDataset(dongting_attack_valid, 1))

# print(f"Added DongTing normal training data: {dongting_normal_train}")
# print(f"Added DongTing attack training data: {dongting_attack_train}")
# print(f"Added DongTing normal validation data: {dongting_normal_valid}")
# print(f"Added DongTing attack validation data: {dongting_attack_valid}")

# Create balanced datasets by undersampling normal data
def create_balanced_datasets(normal_path: str, attack_path: str) -> Tuple[Subset, H5LazyDataset]:
    """Create balanced datasets by undersampling the majority class."""
    # Load full datasets
    normal_dataset = H5LazyDataset(normal_path, 0)
    attack_dataset = H5LazyDataset(attack_path, 1)
    
    normal_count = len(normal_dataset)
    attack_count = len(attack_dataset)
    
    print(f"Original counts: {normal_count} normal, {attack_count} attack (ratio {normal_count/attack_count:.1f}:1)")
    
    # Balance to 2:1 ratio
    target_normal_count = attack_count * 2 
    target_attack_count = attack_count     
    
    print(f"Balanced counts: {target_normal_count} normal, {target_attack_count} attack (ratio 2:1)")
    
    # Create random subset of normal data
    random.seed(42)  # For reproducibility
    normal_indices = random.sample(range(normal_count), min(target_normal_count, normal_count))
    balanced_normal_dataset = Subset(normal_dataset, normal_indices)
    
    return balanced_normal_dataset, attack_dataset

# Add LID-DS datasets 
lid_normal_path = os.path.join(LID_DATA_DIR, "0_normal.h5")
lid_attack_path = os.path.join(LID_DATA_DIR, "1_attack.h5")

# Assert LID-DS directory and files exist
assert LID_DS_AVAILABLE, f"LID-DS dataset directory not found at {LID_DATA_DIR}"
assert os.path.exists(lid_normal_path), f"LID-DS normal data not found at {lid_normal_path}"
assert os.path.exists(lid_attack_path), f"LID-DS attack data not found at {lid_attack_path}"

# Create balanced datasets
balanced_normal_dataset, attack_dataset = create_balanced_datasets(lid_normal_path, lid_attack_path)

# Update dataset creation
train_datasets = [balanced_normal_dataset, attack_dataset]
valid_datasets = [balanced_normal_dataset, attack_dataset]

print("Balanced dataset created:")
print(
    f"   Training: {len(balanced_normal_dataset)} normal + {len(attack_dataset)} attack = "
    f"{len(balanced_normal_dataset) + len(attack_dataset)} total"
)

# Ensure we have datasets to work with
assert train_datasets, "No training datasets found. Please check your data paths."
assert valid_datasets, "No validation datasets found. Please check your data paths."

# Combine all datasets
train_dataset: ConcatDataset = ConcatDataset(train_datasets)
valid_dataset: ConcatDataset = ConcatDataset(valid_datasets)

print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(valid_dataset)}")

cpu_count = os.cpu_count()
num_workers = (cpu_count // 2) if cpu_count else 1
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
    num_workers=num_workers
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate,
    num_workers=num_workers
)

# ====================
# Model instantiation
# ====================

model = LSTMClassifier(LSTMConfig(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    lr=LEARNING_RATE
))

# ====================
# Training
# ====================

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_f1",
    mode="max",
    save_top_k=1,
    filename="best-val-f1",
    verbose=True
)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor="val_f1",
    mode="max",
    patience=3,
    min_delta=0.001,
    verbose=True
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    logger=False,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback, early_stop_callback],
)
trainer.fit(model, train_loader, valid_loader)
