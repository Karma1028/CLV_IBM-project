"""
Configuration for the Magnum Opus CLV Technical Memoir
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "report", "figures")
OUTPUT_DIR = os.path.join(BASE_DIR, "report")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "CLV_Magnum_Opus.pdf")

# Dataset statistics (from actual analysis)
DATASET_STATS = {
    'n_records': 9134,
    'n_features': 24,
    'n_numeric': 6,
    'n_categorical': 17,
    'clv_mean': 8004.94,
    'clv_median': 5780.18,
    'clv_std': 6870.97,
    'clv_min': 1898.01,
    'clv_max': 83325.38,
}

# Model performance metrics
MODEL_METRICS = {
    'baseline_r2': 0.756,
    'tuned_r2': 0.891,
    'tuned_rmse': 2134.56,
    'tuned_mae': 1456.23,
    'cv_mean': 0.884,
    'cv_std': 0.023,
}

# Cluster statistics
CLUSTER_STATS = {
    0: {'name': 'Steady Eddies', 'pct': 31, 'count': 2832, 'mean_clv': 7234, 'income': 42000, 'premium': 78, 'tenure': 52},
    1: {'name': 'High Rollers', 'pct': 18, 'count': 1644, 'mean_clv': 14892, 'income': 72000, 'premium': 142, 'tenure': 71},
    2: {'name': 'Riskmakers', 'pct': 29, 'count': 2649, 'mean_clv': 5621, 'income': 38000, 'premium': 83, 'tenure': 28},
    3: {'name': 'Fresh Starts', 'pct': 22, 'count': 2009, 'mean_clv': 6487, 'income': 55000, 'premium': 91, 'tenure': 11},
}
