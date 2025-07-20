"""GAN module to generate synthetic syscall sequences for testing model generalization."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import os
import pytorch_lightning as pl
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.lstm.trainer import H5LazyDataset, collate

# Hyperparameters for GAN 

NOISE_DIM = 100
SEQ_LEN = 512
VOCAB_SIZE = 600
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
LEARNING_RATE = 2e-4
DISC_LEARNING_RATE = 1e-7 
BATCH_SIZE = 32
MAX_EPOCHS = 25
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4

class Generator(nn.Module):
    """Converts random noise to fake syscall sequences."""
    def __init__(self) -> None:
        """Initialize the Generator network for synthetic syscall sequence generation."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, SEQ_LEN * VOCAB_SIZE),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Noise [batch, 100] to syscall_probs [batch, 512, 600]."""
        out = self.net(noise)
        return out.view(-1, SEQ_LEN, VOCAB_SIZE)

class Discriminator(nn.Module):
    """Tells real syscall sequences from fake ones."""
    def __init__(self) -> None:
        """Initialize the Discriminator network for distinguishing real and fake syscall sequences."""
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM, HIDDEN_DIM * 2, batch_first=True, num_layers=2, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(HIDDEN_DIM, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Sequences [batch, 512] to probability_real [batch, 1]."""
        if sequences.dim() == 3:  # Convert soft probs to IDs
            sequences = torch.multinomial(sequences.view(-1, sequences.size(-1)), 1)
            sequences = sequences.view(-1, sequences.size(1))
        
        embedded = self.embedding(sequences.long())
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(hidden[-1])

def generate_synthetic_h5_files(gan_model: "SyscallGAN", output_dir: str = "../../dataparse/dongting") -> None:
    """Generate synthetic H5 test files to evaluate model generalization."""
    # Generate synthetic test data
    synthetic_normal = gan_model.generate_samples(num_samples=500)
    synthetic_attack = gan_model.generate_samples(num_samples=500)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate test files to evaluate generalization
    synthetic_files = {
        "Synthetic_Normal_DTDS-test.h5": synthetic_normal,       
        "Synthetic_Attach_DTDS-test.h5": synthetic_attack,
    }
    
    # Save each file
    for filename, sequences in synthetic_files.items():
        filepath = os.path.join(output_dir, filename)
        
        with h5py.File(filepath, "w") as f:
            # Create sequences dataset (main data)
            f.create_dataset("sequences", data=sequences, compression="gzip")
            
            # Add metadata
            f.attrs["num_sequences"] = len(sequences)
            f.attrs["sequence_length"] = sequences.shape[1]
            f.attrs["synthetic"] = True
            f.attrs["purpose"] = "generalization_test"

class SyscallGAN(pl.LightningModule):
    """Main GAN class that trains Generator vs Discriminator."""
    def __init__(self) -> None:
        """Initialize the SyscallGAN with generator and discriminator networks."""
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False
    
    def configure_optimizers(self) -> list:
        """Set up optimizers for both networks."""
        gen_opt = optim.Adam(self.generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        disc_opt = optim.Adam(self.discriminator.parameters(), lr=DISC_LEARNING_RATE, betas=(0.5, 0.999))
        return [gen_opt, disc_opt]
    
    def training_step(self, batch: tuple) -> None:
        """Manual optimization: Train both Generator and Discriminator."""
        real_sequences, _, _ = batch  # Use collate format from trainer
        batch_size = real_sequences.size(0)
        
        gen_opt, disc_opt = self.optimizers()
        
        # Train Generator
        gen_opt.zero_grad()
        noise = torch.randn(batch_size, NOISE_DIM, device=self.device)
        fake_sequences = self.generator(noise)
        fake_validity = self.discriminator(fake_sequences)
        gen_loss = self.criterion(fake_validity, torch.ones_like(fake_validity) * 0.9)  # label smoothing
        self.manual_backward(gen_loss)
        
        # Manual gradient clipping for generator
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        gen_opt.step()
        
        # Train Discriminator
        disc_opt.zero_grad()
        real_validity = self.discriminator(real_sequences)
        real_loss = self.criterion(real_validity, torch.ones_like(real_validity) * 0.9)
        
        fake_sequences = self.generator(torch.randn(batch_size, NOISE_DIM, device=self.device))
        fake_validity = self.discriminator(fake_sequences.detach())
        fake_loss = self.criterion(fake_validity, torch.zeros_like(fake_validity) + 0.1)
        
        disc_loss = (real_loss + fake_loss) / 2
        self.manual_backward(disc_loss)
        
        # Manual gradient clipping for discriminator
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        disc_opt.step()
        
        # Log metrics
        self.log("gen_loss", gen_loss, prog_bar=True)
        self.log("disc_loss", disc_loss, prog_bar=True)
        self.log(
            "disc_acc",
            (
                (real_validity > 0.5).float().mean() +
                (fake_validity < 0.5).float().mean()
            ) / 2,
            prog_bar=True
        )
    
    def generate_samples(self, num_samples: int = 100) -> "np.ndarray":
        """Generate synthetic syscall sequences for testing."""
        self.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, NOISE_DIM, device=self.device)
            fake_sequences = self.generator(noise)
            discrete_sequences = torch.multinomial(fake_sequences.view(-1, VOCAB_SIZE), 1)
            discrete_sequences = discrete_sequences.view(num_samples, SEQ_LEN)
        return discrete_sequences.cpu().numpy()

class GenerateH5Callback(pl.Callback):
    """Callback to generate synthetic H5 files after GAN training ends."""

    def on_train_end(self, trainer: pl.Trainer, pl_module: "SyscallGAN") -> None:
        """Generate H5 files when training ends (even if early stopped)."""
        print("Generating synthetic H5 files...")
        generate_synthetic_h5_files(pl_module)

def train_gan(data_dir: str = "../../dataparse/dongting") -> tuple:
    """Train the GAN using existing dataset infrastructure."""
    # Use .h5 files from DongTing loader
    h5_files = [
        os.path.join(data_dir, "Normal_DTDS-train.h5"),
        os.path.join(data_dir, "Attach_DTDS-train.h5")
    ]
    
    datasets = [H5LazyDataset(f, 0) for f in h5_files if os.path.exists(f)]
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    dataloader = DataLoader(
        combined_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 1
    )
    
    # Initialize model and trainer
    gan = SyscallGAN()
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="gen_loss", mode="min", save_top_k=1),
            pl.callbacks.EarlyStopping(
                monitor="gen_loss",
                patience=EARLY_STOP_PATIENCE,
                min_delta=EARLY_STOP_MIN_DELTA
            ),
            GenerateH5Callback()
        ],
    )
    
    trainer.fit(gan, dataloader)
    
    # Get synthetic data after training
    synthetic_data = gan.generate_samples(500)
    
    return gan, synthetic_data  # Return both values

if __name__ == "__main__":
    gan, synthetic_data = train_gan()