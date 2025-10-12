"""Training script for an unsupervised GNN Autoencoder model to be used for the Argos HIDS project."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F # noqa: N812
import numpy as np
from typing import Tuple, Dict
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

from lib.data import load_dataset, CustomGraphDataset
from lib.models import GNNModel
from lib.utils import set_random_seeds

SAVE_MODEL_PATH = "gnn-autoencoder.pt"

class GNNAutoencoder(nn.Module):
    """
    Graph Neural Network Autoencoder for unsupervised anomaly detection on graph-structured data.

    This model uses a GNN-based encoder to generate graph-level embeddings and a feedforward decoder
    to reconstruct node features from these embeddings. The reconstruction error can be used to
    identify anomalies in the input graphs.

    Attributes:
        encoder (GNNModel): GNN-based encoder for graph embeddings.
        decoder (nn.Sequential): Feedforward decoder for reconstructing node features.
    """
    def __init__(
        self,
        vocab_size: int,
        vocab: Dict[int, int],
        embedding_dim: int,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        model_type: str = "GIN"
    ) -> None:
        """
        Initializes the GNNAutoencoder model.

        Args:
            vocab_size (int): Size of the vocabulary for node features.
            vocab (int): The vocabulary itself for node features.
            embedding_dim (int): Dimension of the node embeddings.
            in_channels (int): Number of input node features.
            hidden_channels (int): Number of hidden units in the encoder and decoder.
            num_layers (int): Number of GNN layers in the encoder.
            dropout (float): Dropout rate applied in the encoder.
            model_type (str, optional): Type of GNN model to use in the encoder (e.g., "GIN", "GCN", "SAGE").
                                        Defaults to "GIN".
        """
        super(GNNAutoencoder, self).__init__()

        self.threshold = 0.0 # will be overwritten after training

        self.encoder = GNNModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout,
            act="relu",
            model_type=model_type,
            vocab=vocab
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, in_channels)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass of the GNNAutoencoder.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch vector, which assigns each node to a specific graph.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - The reconstructed node features.
                - The graph-level embeddings.
        """
        # Encode graph-level embeddings
        graph_embeddings = self.encoder(x, edge_index, batch)
        node_embeddings = graph_embeddings[batch]
        reconstructed = self.decoder(node_embeddings)
        return reconstructed, graph_embeddings
    
    def predict(self, sequence: torch.Tensor) -> bool:
        """
        Classifies the given syscall sequence, represented as a PyTorch tensor.

        Args:
            sequence (torch.Tensor): The unidimensional sequence of syscall IDs for the model to classify.
                                     The IDs are already mapped to the values expected by the model.
        
        Returns:
            bool: True if the sequence is malicious; False otherwise.
        """
        if sequence.dim() != 1:
            raise ValueError("Input sequence must be a 1D tensor of syscall IDs.")
        device = next(self.parameters()).device
        with torch.no_grad():
            # reconstruct the sequence
            graph, _ = self.encoder.encode(sequence.tolist())
            batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)
            reconstructed, _ = self.forward(
                graph.x.to(device),
                graph.edge_index.to(device),
                batch
            )

            # calculate the reconstruction error
            node_errors = F.mse_loss(
                reconstructed,
                graph.x.to(device),
                reduction="none"
            ).mean(dim=1)
            graph_error = node_errors.mean().item()

            # malicious if the reconstruction error exceeds the threshold
            return graph_error > self.threshold

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Trains the model for one epoch on the training dataset.

    Args:
        model (nn.Module): The GNN autoencoder model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to run the training on (CPU or CUDA).

    Returns:
        float: The average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # forward pass
        reconstructed, _ = model(batch.x, batch.edge_index, batch.batch)

        # loss computation
        loss = F.mse_loss(reconstructed, batch.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Evaluates the model for one epoch on the validation dataset.

    Args:
        model (nn.Module): The GNN autoencoder model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the evaluation on (CPU or CUDA).

    Returns:
        float: The average validation loss over the epoch.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            reconstructed, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(reconstructed, batch.x)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def calibrate_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    target_fpr: float = 0.05
) -> float:
    """
    Calibrates the anomaly detection threshold based on the validation set to
    achieve a target false positive rate (FPR).

    Args:
        model (nn.Module): The trained GNN autoencoder model.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the evaluation on (CPU or CUDA).
        target_fpr (float, optional): Desired false positive rate for threshold calibration.
                                      Defaults to 0.05.

    Returns:
        float: The calibrated reconstruction error threshold.
    """
    model.eval()
    graph_reconstruction_errors = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            reconstructed, _ = model(batch.x, batch.edge_index, batch.batch)
            node_errors = F.mse_loss(reconstructed, batch.x, reduction="none").mean(dim=1)
            
            # Aggregate node errors to graph level (same as in evaluate_on_test)
            unique_graphs = torch.unique(batch.batch)
            for graph_id in unique_graphs:
                graph_mask = (batch.batch == graph_id)
                graph_error = node_errors[graph_mask].mean().item()
                graph_reconstruction_errors.append(graph_error)
        
    threshold = np.percentile(graph_reconstruction_errors, (1 - target_fpr) * 100)
    print(f"Calibrated threshold at FPR {target_fpr}: {threshold}")
    return float(threshold)

def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float,
    device: torch.device
) -> None:
    """
    Evaluates the trained model on the test dataset and prints the AUC and confusion matrix.

    Args:
        model (nn.Module): The trained GNN autoencoder model.
        test_loader (DataLoader): DataLoader for the test dataset.
        threshold (float): Reconstruction error threshold for anomaly detection.
        device (torch.device): Device to run the evaluation on (CPU or CUDA).
    """
    model.eval()
    graph_reconstruction_errors = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reconstructed, _ = model(batch.x, batch.edge_index, batch.batch)
            node_errors = F.mse_loss(reconstructed, batch.x, reduction="none").mean(dim=1)
            unique_graphs = torch.unique(batch.batch)
            for i, graph_id in enumerate(unique_graphs):
                graph_mask = (batch.batch == graph_id)
                graph_error = node_errors[graph_mask].mean().item()
                graph_reconstruction_errors.append(graph_error)
                graph_label = batch.y[i].item()
                labels.append(graph_label)
    
    # predictions: error > threshold = anomaly
    predictions = (np.array(graph_reconstruction_errors) > threshold).astype(int)

    auc = roc_auc_score(labels, graph_reconstruction_errors)
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions))

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for configuring and running the GNN Autoencoder training and evaluation.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments, including dataset paths,
                            model parameters, and training parameters.
    """
    parser = argparse.ArgumentParser(description="Train a GNN Autoencoder for anomaly detection")
    
    # Dataset paths
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to the training dataset (e.g., DongTing)")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Path to the test dataset (e.g., ADFA)")
    
    # Model parameters
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--hidden_channels", type=int, default=256,
                        help="Number of hidden channels")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    parser.add_argument("--model_type", type=str, default="GIN", choices=["GIN", "GCN", "SAGE"],
                        help="Type of GNN model")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--target_fpr", type=float, default=0.05,
                        help="Target false positive rate for threshold calibration")
    
    return parser.parse_args()

def main() -> None:
    """Main (entrypoint) function for thre training script."""
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds()

    print("Loading datasets...")
    train_data, vocab_size, vocab = load_dataset(args.train_dataset)
    test_data, _, _ = load_dataset(args.test_dataset)

    train_graphs = [d["graph"] for d in train_data if d["label"] == "normal"]
    train_size = int(0.8 * len(train_graphs))
    
    train_dataset = CustomGraphDataset(train_graphs[:train_size], classes=2, training=True)
    val_dataset = CustomGraphDataset(train_graphs[train_size:], classes=2, training=False)
    test_graphs = []
    for d in test_data:
        graph = d["graph"]
        label = 0 if d["label"] == "normal" else 1
        graph.y = torch.tensor([label], dtype=torch.long)
        test_graphs.append(graph)
    test_dataset = CustomGraphDataset(test_graphs, classes=2, training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Initializing model...")
    model = GNNAutoencoder(
        vocab_size=vocab_size,
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        in_channels=train_dataset[0].num_node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print("Calibrating threshold...")
    threshold = calibrate_threshold(model, val_loader, device, target_fpr=args.target_fpr)

    print("Evaluating on test dataset...")
    evaluate_on_test(model, test_loader, threshold, device)

    # save the model
    model.threshold = threshold
    model.eval()
    torch.save(model, SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    main()