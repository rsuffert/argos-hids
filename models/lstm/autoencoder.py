"""
Module for training an LSTM Autoencoder for syscall anomaly detection on normal sequences only.
Uses only normal syscall sequences to learn patterns and detect anomalies through reconstruction error.
"""

import os
from typing import Tuple, List
from dataclasses import dataclass
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import h5py
import numpy as np

torch.backends.cudnn.enabled = False

# =================
# Hyperparameters
# =================

INPUT_SIZE = 1              # Each element in the sequence is a scalar
HIDDEN_SIZE = 64            # Number of hidden units in the LSTM
NUM_LAYERS = 2              # Number of stacked LSTM layers
LEARNING_RATE = 1e-3        # Learning rate for the optimizer
BATCH_SIZE = 64             # Batch size for DataLoader
MAX_EPOCHS = 40             # Number of training epochs
MAX_SEQ_LEN = 2048          # Maximum sequence length for padding/truncation
EARLY_STOP_PATIENCE = 5     # Patience for early stopping training
EARLY_STOP_MIN_DELTA = 50   # Minimum change to qualify as an improvement
THRESHOLD_PERCENTILE = 80.0 # Percentile of reconstruction errors to use as anomaly detection threshold

# ==========================
# Dataset (H5 files) paths
# ==========================

DT_BASE_DIR = os.path.join("..", "..", "dataparse", "dongting")
NORMAL_TRAIN_DT_PATH = os.getenv("NORMAL_TRAIN_DT_PATH",
    os.path.join(DT_BASE_DIR, "Normal_DTDS-train.h5"))
NORMAL_VALID_DT_PATH = os.getenv("NORMAL_VALID_DT_PATH",
    os.path.join(DT_BASE_DIR, "Normal_DTDS-validation.h5"))

# ============================
# Training utilities & logic
# ============================

def collate(batch: List[Tuple[np.ndarray, int]]) -> Tuple[Tensor, Tensor]:
    """Custom collate function to pad sequences and prepare batches."""
    sequences, _ = map(list, zip(*batch, strict=False))
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
    return padded_sequences, lengths

@dataclass
class LSTMAutoencoderConfig:
    """Configuration for the LSTM autoencoder."""
    input_size: int = INPUT_SIZE
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    lr: float = LEARNING_RATE

class LSTMAutoencoder(pl.LightningModule):
    """LSTM autoencoder for anomaly detection."""

    def __init__(self, config: LSTMAutoencoderConfig) -> None:
        """Initialize the LSTM autoencoder."""
        super().__init__()
        self.save_hyperparameters(config.__dict__)
        self.lr = config.lr
        self.hidden_size = config.hidden_size
        self.threshold = torch.jit.Attribute(0.0, float) # will be overwritten

        # Encoder: takes syscall IDs
        self.encoder = torch.nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )

        # Decoder: takes encoder's hidden states
        self.decoder = torch.nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )

        self.output_layer = torch.nn.Linear(config.hidden_size, config.input_size)
        self.criterion = torch.nn.MSELoss() # to measure how well the autoencoder reconstructs the input

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass through encoder-decoder autoencoder."""
        packed_input = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.encoder(packed_input)

        batch_size, seq_len = x.size(0), x.size(1)
        decoder_input = torch.zeros(
            batch_size, seq_len, self.hidden_size, device=x.device
        )
        decoded, _ = self.decoder(decoder_input, (h_n, c_n))

        return self.output_layer(decoded)


    def shared_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Shared logic for training and validation steps."""
        sequences, lengths = batch
        sequences = sequences.unsqueeze(-1)
        reconstructed = self(sequences, lengths)

        # accumulate reconstruction loss for each sequence in the batch (ignoring padding)
        losses = []
        batch_size = sequences.size(0)
        for i in range(batch_size):
            seq_len = int(lengths[i].item()) # ignoring padding
            original_seq = sequences[i, :seq_len, :]
            reconst_seq = reconstructed[i, :seq_len, :]
            loss_i = self.criterion(reconst_seq, original_seq)
            losses.append(loss_i)

        return torch.stack(losses).mean()

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step for one batch."""
        loss = self.shared_step(batch)

        # debug logging every 100 batches
        if batch_idx % 100 == 0:
            sequences, lengths = batch
            sequences = sequences.unsqueeze(-1)
            reconstructed = self(sequences, lengths)
            
            seq_len = int(lengths[0].item())
            orig_sample = sequences[0, :seq_len, 0].cpu().numpy()
            recon_sample = reconstructed[0, :seq_len, 0].detach().cpu().numpy()
            
            print(f"Batch {batch_idx}")
            print(f"Original:     {orig_sample[:10]}")  # First 10 syscalls
            print(f"Reconstructed: {recon_sample[:10]}")
            print(f"Loss: {loss:.4f}")

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], _batch_idx: int) -> None:
        """Validation step for one batch."""
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    @torch.jit.export
    def reconstruction_error(self, sequence: torch.Tensor) -> float:
        """
        Computes the reconstruction MSE for a single sequence.
        Make sure to set the model to eval mode before calling this.

        Args:
            sequence (torch.Tensor): The sequence to compute MSE for.

        Returns:
            float: The MSE for the given sequence.
        """
        if sequence.dim() != 1:
            raise ValueError("Input sequence must be a 1D tensor of syscall IDs.")
        seq_len = sequence.shape[0]
        sequence = sequence.unsqueeze(0).unsqueeze(-1) # shape (1, seq_len, 1)
        lengths = torch.tensor([seq_len], dtype=torch.long, device=sequence.device)
        with torch.no_grad():
            reconstructed = self.forward(sequence, lengths)
        return torch.mean((sequence - reconstructed) ** 2).item()
    
    @torch.jit.export
    def predict(self, sequence: torch.Tensor) -> bool:
        """
        Classifies the given syscall sequence, represented as a PyTorch tensor.
        Make sure to set the model to eval mode before calling this.

        Args:
            sequence (torch.Tensor): The unidimensional sequence of syscall IDs for the model to classify.
                                    The IDs are already mapped to the values expected by the model.
        
        Returns:
            bool: True if the sequence is malicious; False otherwise.
        """
        return self.reconstruction_error(sequence) > self.threshold # type: ignore[operator]
    
    @torch.jit.export
    def set_threshold(self, threshold: float) -> None:
        """
        Set the reconstruction error threshold for anomaly detection.

        Args:
            threshold (float): The threshold value above which a sequence is considered anomalous.
        """
        self.threshold = threshold # type: ignore[assignment]

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

def sanity_check() -> None:
    """Quick sanity check to verify the model is correctly configured before training."""
    print("========= Quick Sanity Check =========")
    
    dummy_sequences = torch.tensor([
        # sequences padded with -1
        [1, 2, 3, 45, 89, -1, -1, -1],
        [12, 56, 78, -1, -1, -1, -1, -1]
    ], dtype=torch.float32)
    dummy_lengths = torch.tensor([5, 3]) # ignoring padding

    model = LSTMAutoencoder(LSTMAutoencoderConfig())
    model.eval()

    with torch.no_grad():
        dummy_sequences = dummy_sequences.unsqueeze(-1) # add feature dimension
        output = model(dummy_sequences, dummy_lengths)

        assert dummy_sequences.shape == output.shape, "Expected output shape to match input shape!"

        assert not torch.isnan(output).any(), "Output contains NaN values!"
        assert not torch.isinf(output).any(), "Output contains Inf values!"
        print("✅ No NaN/Inf values detected")

        for i in range(len(dummy_sequences)):
            seq_len = dummy_lengths[i]
            input_seq = dummy_sequences[i, :seq_len, 0]
            output_seq = output[i, :seq_len, 0]

            unique_outputs = torch.unique(output_seq)
            assert len(unique_outputs) > 1, f"Sequence {i}: Output stuck on single value {output_seq[0].item()}"

            print(f"✅ Sequence {i+1} (len {seq_len}): Input {input_seq.tolist()}")
            print(f"    Reconstructed: {output_seq.tolist()}")
        
        print("========= All Sanity Checks Passed! =========\n")

def compute_threshold(
    model: LSTMAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    percentile: float = 95.0
) -> float:
    """
    Computes the reconstruction error threshold from a dataloader.

    Args:
        model (LSTMAutoencoder): The trained autoencoder model.
        dataloader (torch.utils.data.DataLoader): DataLoader with normal validation sequences.
        percentile (float): The percentile to use for threshold selection (default: 95.0).

    Returns:
        float: The computed threshold value.
    """
    model.eval()
    errors: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            sequences, lengths = batch
            for i in range(sequences.size(0)):
                seq_len = int(lengths[i].item())
                seq = sequences[i, :seq_len]
                errors.append(model.reconstruction_error(seq))
    errors_array = np.array(errors)
    threshold = np.percentile(errors_array, percentile)
    return float(threshold)

def main() -> None:
    """Main function to train the LSTM autoencoder."""
    sanity_check()

    # Lazily load the training and validation datasets
    train_dataset = H5LazyDataset(NORMAL_TRAIN_DT_PATH, 0)
    valid_dataset = H5LazyDataset(NORMAL_VALID_DT_PATH, 0)

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
    model = LSTMAutoencoder(LSTMAutoencoderConfig())

    # Define callbacks to customize training behavior
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-autoencoder",
        verbose=True
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
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

    threshold = compute_threshold(model, valid_loader, THRESHOLD_PERCENTILE)
    print(f"Setting threshold to {threshold}")
    model.set_threshold(threshold)
    torch.jit.script(model).save("lstm-autoencoder.pt")

if __name__ == "__main__":
    main()