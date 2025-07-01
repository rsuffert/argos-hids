"""
Module for training an LSTM intrusion detection model on the DongTing dataset using
PyTorch Lightning.
"""

import os
from typing import Tuple, List
from dataclasses import dataclass
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchmetrics import F1Score, Accuracy
import h5py
import numpy as np

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
        self.criterion = torch.nn.CrossEntropyLoss()
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

DONGTING_BASE_DIR = os.path.join("..", "..", "datasets", "dongting")

assert os.path.exists(DONGTING_BASE_DIR), f"'{DONGTING_BASE_DIR}' not found"
assert os.path.isdir(DONGTING_BASE_DIR), f"'{DONGTING_BASE_DIR}' not a directory"

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

train_dataset: ConcatDataset = ConcatDataset([
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Normal_DTDS-train.h5"), 0),
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Attach_DTDS-train.h5"), 1)
])
valid_dataset: ConcatDataset = ConcatDataset([
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Normal_DTDS-validation.h5"), 0),
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Attach_DTDS-validation.h5"), 1),
])

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
