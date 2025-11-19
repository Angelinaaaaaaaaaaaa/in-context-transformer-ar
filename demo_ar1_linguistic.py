"""
Demo script: AR(1) linguistic baseline replication.

This replicates the key finding from Sander et al. (2024) that linguistic
structure (preserving word order) is better explained by AR(1) dynamics
than shuffled control sequences.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from data import load_moby_dick_embeddings, fit_ar1_to_linguistic_data
from analysis import plot_ar1_linguistic_comparison

print("="*60)
print("AR(1) LINGUISTIC BASELINE REPLICATION")
print("="*60)
print("\nThis demo replicates the finding from Sander et al. (2024)")
print("that linguistic text is better modeled by AR(1) than random sequences.\n")

# Load GPT-2 embeddings from Moby Dick
print("1. Loading Moby Dick text and extracting GPT-2 embeddings...")
print("   (This may take a few minutes on first run)")

try:
    embeddings, tokens = load_moby_dick_embeddings(
        max_tokens=5000,  # Use 5000 tokens for demo
        cache_path='moby_dick_embeddings.npz',
        device='cpu'
    )
    print(f"   ✓ Loaded {len(embeddings)} token embeddings")
    print(f"   ✓ Embedding dimension: {embeddings.shape[1]}")
    print(f"   ✓ First 5 tokens: {tokens[:5]}")
except Exception as e:
    print(f"   ✗ Error loading embeddings: {e}")
    print("\n   Note: This demo requires NLTK and transformers.")
    print("   Install with: pip install nltk transformers")
    print("   Then run: python -c \"import nltk; nltk.download('gutenberg')\"")
    exit(1)

# Fit AR(1) to ordered vs. shuffled sequences
print("\n2. Fitting AR(1) models to linguistic vs. shuffled sequences...")
T = 5  # Sequence length
n_samples = 500  # Number of sequences to fit

losses_ordered, losses_shuffled = fit_ar1_to_linguistic_data(
    embeddings=embeddings,
    T=T,
    n_samples=n_samples,
    seed=42
)

print(f"   ✓ Fitted {n_samples} AR(1) models to ordered sequences")
print(f"   ✓ Fitted {n_samples} AR(1) models to shuffled sequences")

# Compare losses
print("\n3. Comparing fitting losses...")
print(f"   Ordered (linguistic) mean loss: {losses_ordered.mean():.6f}")
print(f"   Shuffled (control) mean loss:   {losses_shuffled.mean():.6f}")
print(f"   Ratio (shuffled / ordered):     {losses_shuffled.mean() / losses_ordered.mean():.4f}×")

if losses_ordered.mean() < losses_shuffled.mean():
    print(f"\n   ✓ Linguistic structure is better explained by AR(1)!")
    print(f"     This confirms the finding from Sander et al. (2024)")
else:
    print(f"\n   ✗ Unexpected: shuffled sequences have lower AR(1) loss")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_ind(losses_ordered, losses_shuffled)
print(f"\n   t-test: t = {t_stat:.4f}, p = {p_value:.4e}")

if p_value < 0.001:
    print(f"   ✓ Highly significant difference (p < 0.001)")

# Generate plot
print("\n4. Generating visualization...")
plot_ar1_linguistic_comparison(
    losses_ordered,
    losses_shuffled,
    save_path='ar1_linguistic_comparison.png'
)
print(f"   ✓ Saved plot to: ar1_linguistic_comparison.png")

print("\n" + "="*60)
print("DEMO COMPLETED!")
print("="*60)
print("\nKey finding: Linguistic text exhibits temporal structure that")
print("is well-captured by AR(1) dynamics, validating the use of AR")
print("models for studying in-context learning on language data.")
