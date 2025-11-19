"""Data generation and loading utilities."""
from .ar_process import (
    generate_stable_ar_weights,
    generate_ar_sequence,
    generate_ar_dataset,
    compute_ar_fit_loss,
    check_stability,
    companion_matrix
)
from .gpt2_embeddings import (
    extract_gpt2_embeddings,
    load_moby_dick_embeddings,
    create_ar_dataset_from_embeddings,
    fit_ar1_to_linguistic_data
)

__all__ = [
    'generate_stable_ar_weights',
    'generate_ar_sequence',
    'generate_ar_dataset',
    'compute_ar_fit_loss',
    'check_stability',
    'companion_matrix',
    'extract_gpt2_embeddings',
    'load_moby_dick_embeddings',
    'create_ar_dataset_from_embeddings',
    'fit_ar1_to_linguistic_data',
]
