"""
Training utilities with early stopping and logging.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
import os


class ARDataset(Dataset):
    """Dataset for AR(p) sequences."""

    def __init__(self, sequences: np.ndarray):
        """
        Args:
            sequences: Array of shape (n_sequences, T, d)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class Trainer:
    """
    Trainer for Transformer model on AR(p) sequences.
    """

    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 context_len: int,
                 lr: float = 3e-4,
                 batch_size: int = 64,
                 max_epochs: int = 100,
                 patience: int = 10,
                 device: str = 'cpu',
                 save_dir: Optional[str] = None):
        """
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            context_len: Length of context window
            lr: Learning rate
            batch_size: Batch size
            max_epochs: Maximum number of epochs
            patience: Early stopping patience
            device: Device to train on
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.context_len = context_len

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.max_epochs = max_epochs
        self.patience = patience
        self.save_dir = save_dir

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []

    def compute_loss(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for a batch of sequences.

        Args:
            sequences: Batch of sequences of shape (batch, T, d)

        Returns:
            Loss value
        """
        batch_size, T, d = sequences.shape

        # Split into context and target
        # Use first context_len as context, predict remaining
        if T <= self.context_len:
            # If sequence is too short, use first T-1 as context
            context = sequences[:, :-1, :]
            target = sequences[:, 1:, :]
        else:
            context = sequences[:, :self.context_len, :]
            target = sequences[:, self.context_len:, :]

        # Forward pass
        predictions, _ = self.model(sequences[:, :-1, :])  # Predict all positions except first

        # Only compute loss on target positions
        if T <= self.context_len:
            pred_target = predictions[:, -target.shape[1]:, :]
        else:
            pred_target = predictions[:, self.context_len-1:, :]

        # Compute MSE loss
        loss = self.criterion(pred_target, target)

        return loss

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for sequences in self.train_loader:
            sequences = sequences.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            loss = self.compute_loss(sequences)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for sequences in self.val_loader:
                sequences = sequences.to(self.device)
                loss = self.compute_loss(sequences)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train(self, verbose: bool = True) -> dict:
        """
        Train model with early stopping.

        Args:
            verbose: If True, print progress

        Returns:
            Training history
        """
        if verbose:
            pbar = tqdm(range(self.max_epochs), desc="Training")
        else:
            pbar = range(self.max_epochs)

        for epoch in pbar:
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}',
                    'best_val': f'{self.best_val_loss:.6f}'
                })

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                if self.save_dir is not None:
                    self.save_checkpoint('best_model.pt')
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        if self.save_dir is not None:
            self.load_checkpoint('best_model.pt')

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'n_epochs': len(self.train_losses)
        }

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.save_dir is None:
            return

        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        if self.save_dir is None:
            return

        path = os.path.join(self.save_dir, filename)
        if not os.path.exists(path):
            print(f"Checkpoint {path} not found")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


if __name__ == "__main__":
    # Test trainer
    print("Testing trainer...")

    from models.transformer import GPTModel
    from data.ar_process import generate_ar_dataset

    # Generate synthetic data
    p, d, T = 2, 5, 100
    context_len = 70

    train_sequences, train_weights = generate_ar_dataset(
        n_sequences=1000, p=p, d=d, T=T, noise_std=0.1, same_dynamics=False, seed=42
    )
    val_sequences, val_weights = generate_ar_dataset(
        n_sequences=200, p=p, d=d, T=T, noise_std=0.1, same_dynamics=False, seed=43
    )

    print(f"Train data shape: {train_sequences.shape}")
    print(f"Val data shape: {val_sequences.shape}")

    # Create datasets
    train_dataset = ARDataset(train_sequences)
    val_dataset = ARDataset(val_sequences)

    # Create model
    model = GPTModel(
        d_input=d,
        d_model=128,  # Smaller for testing
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=T,
        dropout=0.1
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        context_len=context_len,
        lr=3e-4,
        batch_size=32,
        max_epochs=5,  # Few epochs for testing
        patience=3,
        device='cpu'
    )

    # Train
    history = trainer.train(verbose=True)

    print(f"\nTraining completed:")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Number of epochs: {history['n_epochs']}")
