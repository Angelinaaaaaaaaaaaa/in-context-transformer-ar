"""
Quick test script to verify that all components work correctly.
"""
import torch
import numpy as np
import os
import sys

print("="*60)
print("TESTING IMPLEMENTATION")
print("="*60)

# Test 1: Data generation
print("\n1. Testing AR(p) data generation...")
from data import generate_ar_dataset, compute_ar_fit_loss

p, d, T = 3, 5, 100
sequences, weights = generate_ar_dataset(
    n_sequences=10, p=p, d=d, T=T, noise_std=0.1,
    same_dynamics=False, seed=42
)
print(f"   ✓ Generated {sequences.shape[0]} sequences of shape {sequences.shape[1:]} for AR({p})")

# Test fitting
loss, fitted_weights = compute_ar_fit_loss(sequences[0], p)
print(f"   ✓ AR({p}) fitting loss: {loss:.6f}")

# Test 2: GPT model
print("\n2. Testing GPT model...")
from models import GPTModel

model = GPTModel(
    d_input=d,
    d_model=128,  # Smaller for testing
    n_layers=2,
    n_heads=4,
    d_ff=512,
    max_seq_len=T,
    dropout=0.1
)
print(f"   ✓ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
x = torch.tensor(sequences[:4], dtype=torch.float32)
output, attention = model(x, return_attention=True)
print(f"   ✓ Forward pass: input {x.shape} -> output {output.shape}")
print(f"   ✓ Attention: {len(attention)} layers, shape {attention[0].shape}")

# Test prediction
context = x[:, :70, :]
next_pred = model.predict_next(context)
print(f"   ✓ Next prediction: {next_pred.shape}")

rollout = model.autoregressive_predict(context, n_steps=10)
print(f"   ✓ 10-step rollout: {rollout.shape}")

# Test 3: Baseline models
print("\n3. Testing baseline models...")
from models import OraclePredictor, OLSPredictor, LastValuePredictor

oracle = OraclePredictor(weights[0])
ols = OLSPredictor(p)
last_val = LastValuePredictor()

oracle_pred = oracle.predict_next(context)
ols_pred = ols.predict_next(context)
last_pred = last_val.predict_next(context)

print(f"   ✓ Oracle prediction: {oracle_pred.shape}")
print(f"   ✓ OLS prediction: {ols_pred.shape}")
print(f"   ✓ Last-value prediction: {last_pred.shape}")

# Test 4: Training (quick test)
print("\n4. Testing training loop...")
from training import Trainer, ARDataset

train_sequences, _ = generate_ar_dataset(
    n_sequences=100, p=2, d=5, T=100, noise_std=0.1, seed=42
)
val_sequences, _ = generate_ar_dataset(
    n_sequences=20, p=2, d=5, T=100, noise_std=0.1, seed=43
)

train_dataset = ARDataset(train_sequences)
val_dataset = ARDataset(val_sequences)

test_model = GPTModel(
    d_input=5, d_model=64, n_layers=2, n_heads=2, d_ff=256, max_seq_len=100
)

trainer = Trainer(
    model=test_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    context_len=70,
    lr=3e-4,
    batch_size=16,
    max_epochs=2,  # Just 2 epochs for testing
    patience=5,
    device='cpu'
)

history = trainer.train(verbose=False)
print(f"   ✓ Training completed: {history['n_epochs']} epochs")
print(f"   ✓ Best val loss: {history['best_val_loss']:.6f}")

# Test 5: Evaluation metrics
print("\n5. Testing evaluation metrics...")
from training import evaluate_model

test_sequences, test_weights = generate_ar_dataset(
    n_sequences=50, p=2, d=5, T=100, noise_std=0.1, seed=44
)
test_sequences_tensor = torch.tensor(test_sequences, dtype=torch.float32)

oracle = OraclePredictor(test_weights[0])
ols = OLSPredictor(2)

metrics = evaluate_model(
    model=test_model,
    test_sequences=test_sequences_tensor,
    test_weights=test_weights,
    context_len=70,
    p=2,
    oracle_predictor=oracle,
    ols_predictor=ols,
    device='cpu'
)

print(f"   ✓ 1-step MSE: {metrics['mse_1step']:.6f}")
print(f"   ✓ 1-step relative error: {metrics['rel_error_1step']:.4f}")
print(f"   ✓ 10-step rollout MSE: {metrics['mse_rollout']:.6f}")

# Test 6: Attention analysis
print("\n6. Testing attention analysis...")
from analysis import analyze_head_specialization, cluster_attention_heads

test_seq = torch.tensor(test_sequences[:10], dtype=torch.float32)

lag_attention, dominant_lags = analyze_head_specialization(
    model=test_model,
    sequences=test_seq,
    layer_idx=-1,
    max_lag=5
)

print(f"   ✓ Lag attention shape: {lag_attention.shape}")
print(f"   ✓ Dominant lags: {dominant_lags}")

cluster_labels, cluster_centers = cluster_attention_heads(
    lag_attention, n_clusters=2
)
print(f"   ✓ Cluster labels: {cluster_labels}")
print(f"   ✓ Cluster centers shape: {cluster_centers.shape}")

# Test 7: Plotting (without displaying)
print("\n7. Testing plotting functions...")
from analysis import plot_scaling_results, set_style
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

set_style()

dummy_results = {
    1: {'mse_1step': 0.1, 'mse_1step_oracle': 0.05, 'mse_1step_ols': 0.12,
        'mse_rollout': 0.2, 'mse_rollout_oracle': 0.1, 'mse_rollout_ols': 0.15,
        'rel_error_1step': 2.0, 'rel_error_rollout': 2.0},
    2: {'mse_1step': 0.15, 'mse_1step_oracle': 0.07, 'mse_1step_ols': 0.18,
        'mse_rollout': 0.3, 'mse_rollout_oracle': 0.14, 'mse_rollout_ols': 0.25,
        'rel_error_1step': 2.14, 'rel_error_rollout': 2.14},
}

# Create temp directory for test plots
os.makedirs('test_plots', exist_ok=True)
plot_scaling_results(dummy_results, save_path='test_plots/test_scaling.png')
print(f"   ✓ Generated test plot: test_plots/test_scaling.png")

# Clean up test plots
import shutil
shutil.rmtree('test_plots')

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nYou can now run experiments:")
print("  - H1 (scaling): python experiments/h1_scaling.py --help")
print("  - H2 (noise): python experiments/h2_noise.py --help")
print("  - H3 (mechanism): python experiments/h3_mechanism.py --help")
print("  - All at once: ./run_all_experiments.sh [cpu|cuda]")
