import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchmetrics import F1Score, Accuracy
import os
import h5py
from typing import List

# =============================
# Hyperparameters and constants
# =============================

INPUT_SIZE = 1              # Each element in the sequence is a scalar
HIDDEN_SIZE = 64            # Number of hidden units in the LSTM
NUM_LAYERS = 2              # Number of stacked LSTM layers
NUM_CLASSES = 2             # Output classes: normal or attack
LEARNING_RATE = 1e-3        # Learning rate for the optimizer
BATCH_SIZE = 128            # Batch size for DataLoader
MAX_EPOCHS = 50              # Number of training epochs
TRAIN_ATTACK_SPLIT = 0.6    # Proportion of attack data used for training

# ====================
# Collate function
# ====================

def collate(batch):
    """Custom collate function to pad sequences and prepare batches."""
    sequences, labels = zip(*batch)
    # PyTorch expects tensors to be floating-point, even though they are scalars in our case
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    # padding sequences with zeros to the maximum length in the batch
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, lengths, labels

# ====================
# Model definition
# ====================

class LSTMClassifier(pl.LightningModule):
    """LSTM-based classifier using PyTorch Lightning."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='binary')
        self.f1 = F1Score(task='binary')

    def forward(self, x, lengths):
        """Forward pass through LSTM and classification layer."""
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        out = self.fc(h_n[-1])
        return out

    def shared_step(self, batch, step_type):
        """Shared logic for training and validation steps."""
        sequences, lengths, labels = batch
        sequences = sequences.unsqueeze(-1)
        outputs = self(sequences, lengths)
        preds = outputs.argmax(dim=1)

        loss = self.criterion(outputs, labels)
        acc = self.accuracy(preds, labels)
        f1 = self.f1(preds, labels)
        self.log(f'{step_type}_loss', loss, prog_bar=(step_type == 'val'))
        self.log(f'{step_type}_acc', acc, prog_bar=(step_type == 'val'))
        self.log(f'{step_type}_f1', f1, prog_bar=(step_type == 'val'))

        return loss

    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        return self.shared_step(batch, step_type='train')

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, step_type='val')

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ====================
# Data loading
# ====================

DONGTING_BASE_DIR = os.path.join("..", "dataset", "dongting")

assert os.path.exists(DONGTING_BASE_DIR), f"DongTing dataset directory not found at '{DONGTING_BASE_DIR}'"
assert os.path.isdir(DONGTING_BASE_DIR), f"Expected a directory at '{DONGTING_BASE_DIR}' but found a file or non-existent path"

class H5LazyDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str, label: int):
        assert os.path.exists(h5_path), f"HDF5 file not found at '{h5_path}'"
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as h5f:
            self.length = len(h5f['sequences'])
        self.label = label

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5f:
            sequence = h5f['sequences'][idx]
        return sequence, self.label

train_dataset = ConcatDataset([
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Normal_DTDS-train.h5"), 0),
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Attach_DTDS-train.h5"), 1)
])
valid_dataset = ConcatDataset([
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Normal_DTDS-validation.h5"), 0),
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Attach_DTDS-validation.h5"), 1),
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Normal_DTDS-test.h5"), 0),
    H5LazyDataset(os.path.join(DONGTING_BASE_DIR, "Attach_DTDS-test.h5"), 1)
])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

# ====================
# Model instantiation
# ====================

model = LSTMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    lr=LEARNING_RATE
)

# ====================
# Training
# ====================

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    logger=False,
    enable_checkpointing=False
)
trainer.fit(model, train_loader, valid_loader)