"""Analysis and visualization utilities."""
from .attention import (
    aggregate_attention_by_lag,
    analyze_head_specialization,
    cluster_attention_heads,
    ablate_attention_heads,
    compute_head_importance,
    analyze_lag_specific_ablation
)
from .plotting import (
    plot_scaling_results,
    plot_noise_robustness,
    plot_attention_heatmap,
    plot_head_clustering,
    plot_ablation_results,
    plot_training_curves,
    plot_ar1_linguistic_comparison,
    set_style
)

__all__ = [
    'aggregate_attention_by_lag',
    'analyze_head_specialization',
    'cluster_attention_heads',
    'ablate_attention_heads',
    'compute_head_importance',
    'analyze_lag_specific_ablation',
    'plot_scaling_results',
    'plot_noise_robustness',
    'plot_attention_heatmap',
    'plot_head_clustering',
    'plot_ablation_results',
    'plot_training_curves',
    'plot_ar1_linguistic_comparison',
    'set_style',
]
