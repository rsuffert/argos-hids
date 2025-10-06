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
import h5py
import numpy as np
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryConfusionMatrix

torch.backends.cudnn.enabled = False

# =================
# Hyperparameters
# =================

INPUT_SIZE = 1              # Each element in the sequence is a scalar
HIDDEN_SIZE = 64            # Number of hidden units in the LSTM
NUM_LAYERS = 2              # Number of stacked LSTM layers
NUM_CLASSES = 2             # Output classes: normal or attack
LEARNING_RATE = 1e-3        # Learning rate for the optimizer
BATCH_SIZE = 64             # Batch size for DataLoader
MAX_EPOCHS = 10             # Number of training epochs
MAX_SEQ_LEN = 2048          # Maximum sequence length for padding/truncation
EARLY_STOP_PATIENCE = 3     # Patience for early stopping training
EARLY_STOP_MIN_DELTA = 1e-3 # Minimum change to qualify as an improvement

# ==========================
# Dataset (H5 files) paths
# ==========================

DT_BASE_DIR = os.path.join("..", "..", "dataparse", "dongting")
NORMAL_TRAIN_DT_PATH = os.getenv("NORMAL_TRAIN_DT_PATH",
    os.path.join(DT_BASE_DIR, "Normal_DTDS-train.h5"))
ATTACK_TRAIN_DT_PATH = os.getenv("ATTACK_TRAIN_DT_PATH",
    os.path.join(DT_BASE_DIR, "Attach_DTDS-train.h5"))
NORMAL_VALID_DT_PATH = os.getenv("NORMAL_VALID_DT_PATH",
    os.path.join(DT_BASE_DIR, "Normal_DTDS-validation.h5"))
ATTACK_VALID_DT_PATH = os.getenv("ATTACK_VALID_DT_PATH",
    os.path.join(DT_BASE_DIR, "Attach_DTDS-validation.h5"))

# ============================
# Training utilities & logic
# ============================

def collate(batch: List[Tuple[np.ndarray, int]]) -> Tuple[Tensor, Tensor, Tensor]:
    """Custom collate function to pad sequences and prepare batches."""
    sequences, labels = map(list, zip(*batch, strict=False))
    # PyTorch expects tensors to be floating-point, even though they are scalars in our case
    sequences = [torch.as_tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32)
                 for seq in sequences]
    lengths = torch.as_tensor(
        [min(len(seq), MAX_SEQ_LEN)
         for seq in sequences],
        dtype=torch.long
    )
    # padding sequences with zeros to the maximum length in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=-1)
    labels_ = torch.as_tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels_

@dataclass
class LSTMConfig:
    """Configuration for the LSTM classifier."""
    input_size: int = INPUT_SIZE
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    num_classes: int = NUM_CLASSES
    lr: float = LEARNING_RATE

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
        self.val_accuracy = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_conf_matrix = BinaryConfusionMatrix()

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass through LSTM and classification layer."""
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed_input)
        return self.fc(h_n[-1])

    def shared_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Shared logic for training and validation steps."""
        sequences, lengths, labels = batch
        sequences = sequences.unsqueeze(-1)
        outputs = self(sequences, lengths)
        preds = outputs.argmax(dim=1)
        loss = self.criterion(outputs, labels)
        return labels, preds, loss

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], _batch_idx: int) -> Tensor:
        """Training step for one batch."""
        _, _, loss = self.shared_step(batch)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], _batch_idx: int) -> None:
        """Validation step for one batch."""
        labels, preds, _ = self.shared_step(batch)
        self.val_f1.update(preds, labels)
        self.val_accuracy.update(preds, labels)
        self.val_conf_matrix.update(preds, labels)

    def on_validation_epoch_end(self) -> None:
        """Callback for the end of validation epoch."""
        cm  = self.val_conf_matrix.compute()
        f1  = self.val_f1.compute()
        acc = self.val_accuracy.compute()
        # PyTorch Lightning's log method only supports floats
        # we're logging this to the configured log file
        self.log("val_TN", float(cm[0, 0]), prog_bar=False)
        self.log("val_FP", float(cm[0, 1]), prog_bar=False)
        self.log("val_FN", float(cm[1, 0]), prog_bar=False)
        self.log("val_TP", float(cm[1, 1]), prog_bar=False)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.val_conf_matrix.reset()
        self.val_f1.reset()
        self.val_accuracy.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict(self, sequence: torch.Tensor) -> bool:
        """
        Classifies the given syscall sequence, represented as a PyTorch tensor.

        Args:
            sequence (torch.Tensor): The unidimensional sequence of syscall IDs for the model to classify.
                                     The IDs are already mapped to the values expected by the model.
        
        Returns:
            bool: True if the sequence is malicious; False otherwise.
        """
        self.eval()
        with torch.no_grad():
            if sequence.dim() != 1:
                raise ValueError("Input sequence must be a 1D tensor of syscall IDs.")
            seq_len = sequence.shape[0]
            # add batch and feature dimensions: (1, seq_len, 1)
            sequence = sequence.unsqueeze(0).unsqueeze(-1)
            lengths = torch.tensor([seq_len], dtype=torch.long, device=sequence.device)
            logits = self.forward(sequence, lengths)
            pred = torch.argmax(logits, dim=1).item()
            return bool(pred)

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
        assert os.path.exists(h5_path), (
            f"H5 file not found at '{h5_path}'. "
            "Did you run the preprocessing script for the dataset?"
        )
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

def main() -> None:
    """Main function to train the LSTM model."""
    # Lazily load the training and validation datasets
    train_dataset: ConcatDataset = ConcatDataset([
        H5LazyDataset(NORMAL_TRAIN_DT_PATH, 0),
        H5LazyDataset(ATTACK_TRAIN_DT_PATH, 1)
    ])
    valid_dataset: ConcatDataset = ConcatDataset([
        H5LazyDataset(NORMAL_VALID_DT_PATH, 0),
        H5LazyDataset(ATTACK_VALID_DT_PATH, 1),
    ])

    # Create DataLoader for training and validation datasets
    # Using half of the available CPU cores for parallel data loading
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

    # Initialize the LSTM model instance with the specified configurations
    model = LSTMClassifier(LSTMConfig())

    # Define callbacks to customize training behavior
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
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        verbose=True
    )

    # Initialize the PyTorch Lightning trainer and start training
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        logger=True,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(model, train_loader, valid_loader)

    torch.jit.script(model).save("lstm.pt")

if __name__ == "__main__":
    main()