"""
Extract GPT-2 token embeddings from text corpus (e.g., Moby Dick).
"""
import numpy as np
import nltk
from transformers import GPT2Tokenizer, GPT2Model
import torch
from typing import Tuple, Optional
import warnings


def download_corpus():
    """Download NLTK Gutenberg corpus if not already available."""
    try:
        nltk.data.find('corpora/gutenberg')
    except LookupError:
        print("Downloading NLTK Gutenberg corpus...")
        nltk.download('gutenberg')


def extract_gpt2_embeddings(text: str,
                           max_tokens: int = 10000,
                           device: str = 'cpu') -> Tuple[np.ndarray, list]:
    """
    Extract GPT-2 token embeddings from text.

    Args:
        text: Input text
        max_tokens: Maximum number of tokens to process
        device: Device to run model on ('cpu' or 'cuda')

    Returns:
        - embeddings: Array of shape (n_tokens, d_model) where d_model=1280 for GPT-2
        - tokens: List of token strings
    """
    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2Model.from_pretrained('gpt2-large')
    model = model.to(device)
    model.eval()

    # Tokenize text
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) > max_tokens:
        encoded = encoded[:max_tokens]

    # Get embeddings
    with torch.no_grad():
        input_ids = torch.tensor([encoded]).to(device)
        # Get token embeddings (not contextual)
        embeddings = model.transformer.wte(input_ids)  # Shape: (1, n_tokens, d_model)
        embeddings = embeddings.squeeze(0).cpu().numpy()  # Shape: (n_tokens, d_model)

    # Decode tokens for reference
    tokens = [tokenizer.decode([token_id]) for token_id in encoded]

    return embeddings, tokens


def create_ar_dataset_from_embeddings(embeddings: np.ndarray,
                                     T: int = 5,
                                     shuffle: bool = False,
                                     seed: Optional[int] = None) -> np.ndarray:
    """
    Create AR(1) dataset from token embeddings by splitting into sequences.

    Args:
        embeddings: Token embeddings of shape (n_tokens, d)
        T: Sequence length
        shuffle: If True, shuffle tokens before creating sequences
        seed: Random seed for shuffling

    Returns:
        Dataset of shape (n_sequences, T, d)
    """
    n_tokens, d = embeddings.shape

    if shuffle and seed is not None:
        np.random.seed(seed)

    if shuffle:
        # Shuffle tokens
        indices = np.random.permutation(n_tokens)
        embeddings = embeddings[indices]

    # Split into sequences
    n_sequences = n_tokens // T
    embeddings = embeddings[:n_sequences * T]
    dataset = embeddings.reshape(n_sequences, T, d)

    return dataset


def fit_ar1_to_linguistic_data(embeddings: np.ndarray,
                               T: int = 5,
                               n_samples: int = 1000,
                               seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit AR(1) model to linguistic data (ordered) and shuffled control.

    This replicates the analysis from Sander et al. (2024) showing that
    linguistic structure is better explained by AR(1) than random sequences.

    Args:
        embeddings: Token embeddings of shape (n_tokens, d)
        T: Sequence length
        n_samples: Number of sequences to sample for fitting
        seed: Random seed

    Returns:
        - losses_ordered: AR(1) fitting losses for ordered sequences
        - losses_shuffled: AR(1) fitting losses for shuffled sequences
    """
    from .ar_process import compute_ar_fit_loss

    if seed is not None:
        np.random.seed(seed)

    # Create ordered and shuffled datasets
    dataset_ordered = create_ar_dataset_from_embeddings(embeddings, T=T, shuffle=False)
    dataset_shuffled = create_ar_dataset_from_embeddings(embeddings, T=T, shuffle=True, seed=seed)

    # Sample sequences
    n_available = min(len(dataset_ordered), len(dataset_shuffled))
    n_samples = min(n_samples, n_available)

    indices = np.random.choice(n_available, n_samples, replace=False)

    # Compute losses
    losses_ordered = []
    losses_shuffled = []

    for idx in indices:
        loss_ord, _ = compute_ar_fit_loss(dataset_ordered[idx], p=1)
        loss_shuf, _ = compute_ar_fit_loss(dataset_shuffled[idx], p=1)
        losses_ordered.append(loss_ord)
        losses_shuffled.append(loss_shuf)

    return np.array(losses_ordered), np.array(losses_shuffled)


def load_moby_dick_embeddings(max_tokens: int = 10000,
                              cache_path: Optional[str] = None,
                              device: str = 'cpu') -> Tuple[np.ndarray, list]:
    """
    Load Moby Dick text and extract GPT-2 embeddings.

    Args:
        max_tokens: Maximum number of tokens to process
        cache_path: Path to cache embeddings (saves time on repeated runs)
        device: Device to run model on

    Returns:
        - embeddings: Array of shape (n_tokens, 1280)
        - tokens: List of token strings
    """
    # Check cache
    if cache_path is not None:
        try:
            data = np.load(cache_path, allow_pickle=True)
            print(f"Loaded cached embeddings from {cache_path}")
            return data['embeddings'], data['tokens'].tolist()
        except FileNotFoundError:
            pass

    # Download corpus if needed
    download_corpus()

    # Load Moby Dick
    print("Loading Moby Dick from NLTK corpus...")
    moby_dick = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')

    # Extract embeddings
    print(f"Extracting GPT-2 embeddings for up to {max_tokens} tokens...")
    embeddings, tokens = extract_gpt2_embeddings(moby_dick, max_tokens, device)

    print(f"Extracted {len(embeddings)} token embeddings of dimension {embeddings.shape[1]}")

    # Cache if requested
    if cache_path is not None:
        np.savez(cache_path, embeddings=embeddings, tokens=np.array(tokens, dtype=object))
        print(f"Cached embeddings to {cache_path}")

    return embeddings, tokens


if __name__ == "__main__":
    # Test embedding extraction
    print("Testing GPT-2 embedding extraction...")

    # Simple test with short text
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    embeddings, tokens = extract_gpt2_embeddings(test_text, max_tokens=100)

    print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    print(f"First 5 tokens: {tokens[:5]}")

    # Test dataset creation
    print("\nTesting dataset creation...")
    dataset = create_ar_dataset_from_embeddings(embeddings, T=5, shuffle=False)
    print(f"Created dataset of shape {dataset.shape}")

    dataset_shuffled = create_ar_dataset_from_embeddings(embeddings, T=5, shuffle=True, seed=42)
    print(f"Created shuffled dataset of shape {dataset_shuffled.shape}")

    # Test AR(1) fitting
    print("\nTesting AR(1) fitting on linguistic data...")
    losses_ordered, losses_shuffled = fit_ar1_to_linguistic_data(
        embeddings, T=5, n_samples=10, seed=42
    )
    print(f"Mean loss (ordered): {losses_ordered.mean():.6f}")
    print(f"Mean loss (shuffled): {losses_shuffled.mean():.6f}")
    print(f"Ordered sequences have {'lower' if losses_ordered.mean() < losses_shuffled.mean() else 'higher'} loss")
