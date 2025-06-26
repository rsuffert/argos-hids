import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchmetrics import F1Score, Accuracy
import numpy as np
import os

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

DONGTING_ABNORMAL_PATH = os.path.join("dataset", "dongting", "syscall_seqs_label_1.npz")
DONGTING_NORMAL_PATH   = os.path.join("dataset", "dongting", "syscall_seqs_label_0.npz")

assert os.path.exists(DONGTING_ABNORMAL_PATH), f"Abnormal data file not found at '{DONGTING_ABNORMAL_PATH}'"
assert os.path.exists(DONGTING_NORMAL_PATH),   f"Normal data file not found at '{DONGTING_NORMAL_PATH}'"

# Load normal and abnormal sequences
abnormal_arr = np.load(DONGTING_ABNORMAL_PATH, allow_pickle=True)["arr_0"]
normal_arr   = np.load(DONGTING_NORMAL_PATH,   allow_pickle=True)["arr_0"]

# Use splits 0, 1, and 3 for training; split 2 for testing
abnormal_train = abnormal_arr[0] + abnormal_arr[1] + abnormal_arr[3]
abnormal_test  = abnormal_arr[2]
normal_train   = normal_arr[0] + normal_arr[1] + normal_arr[3]
normal_test    = normal_arr[2]

# Build final train and test sets
train_sequences = normal_train + abnormal_train
train_labels    = [0]*len(normal_train) + [1]*len(abnormal_train)
test_sequences  = normal_test + abnormal_test
test_labels     = [0]*len(normal_test) + [1]*len(abnormal_test)

# Create PyTorch Datasets instances
class HIDSSyscallsDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = HIDSSyscallsDataset(train_sequences, train_labels)
test_dataset  = HIDSSyscallsDataset(test_sequences,  test_labels)

# Initialize DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

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
trainer.fit(model, train_loader, test_loader)