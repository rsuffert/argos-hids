import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from dataset import ADFALDDataset, load_dataset

# =============================
# Hyperparameters and constants
# =============================

INPUT_SIZE = 1              # Each element in the sequence is a scalar
HIDDEN_SIZE = 64            # Number of hidden units in the LSTM
NUM_LAYERS = 2              # Number of stacked LSTM layers
NUM_CLASSES = 2             # Output classes: normal or attack
LEARNING_RATE = 1e-3        # Learning rate for the optimizer
BATCH_SIZE = 128            # Batch size for DataLoader
MAX_EPOCHS = 4              # Number of training epochs
TRAIN_ATTACK_SPLIT = 0.6    # Proportion of attack data used for training

# ====================
# Collate function
# ====================

def collate(batch):
    """Custom collate function to pad sequences and prepare batches."""
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
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

    def forward(self, x, lengths):
        """Forward pass through LSTM and classification layer."""
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        out = self.fc(h_n[-1])
        return out

    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        sequences, lengths, labels = batch
        sequences = sequences.unsqueeze(-1)
        outputs = self(sequences, lengths)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        sequences, lengths, labels = batch
        sequences = sequences.unsqueeze(-1)
        outputs = self(sequences, lengths)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        print(f"Using Adam optimizer with learning rate: {self.hparams.lr}")
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ====================
# Data loading
# ====================

# Load dataset paths
train_normal_path, valid_normal_path, attack_data_path = load_dataset()

# Initialize datasets
train_normal = ADFALDDataset(train_normal_path, label=0)
valid_normal = ADFALDDataset(valid_normal_path, label=0)
attack_data = ADFALDDataset(attack_data_path, label=1)

# Split attack dataset into train and validation
train_attack_len = int(TRAIN_ATTACK_SPLIT * len(attack_data))
valid_attack_len = len(attack_data) - train_attack_len
train_attack, valid_attack = random_split(attack_data, [train_attack_len, valid_attack_len])

# Concatenate datasets
train_dataset = ConcatDataset([train_normal, train_attack])
valid_dataset = ConcatDataset([valid_normal, valid_attack])

# Initialize DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
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