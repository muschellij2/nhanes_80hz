#!/bin/bash
sbatch code/shell/conda_ssl_wearables.sh --partition=gpu --gpus=1


# Patient Level

# Basic training
# python patient_level_classifier.py /path/to/features /path/to/labels.csv
#
# # Compare aggregation methods
# python patient_level_classifier.py /path/to/features /path/to/labels.csv --compare_methods
#
# # Use logistic regression with hyperparameter optimization
# python patient_level_classifier.py /path/to/features /path/to/labels.csv --classifier logistic_regression --optimize



# Window Level

# # Basic training
# python window_level_classifier.py /path/to/features /path/to/labels.csv
#
# # Compare strategies and classifiers
# python window_level_classifier.py /path/to/features /path/to/labels.csv --compare_strategies --compare_classifiers
#
# # Train deep learning model
# python window_level_classifier.py /path/to/features /path/to/labels.csv --classifier deep --epochs 100
#
# # Full analysis with pattern investigation
# python window_level_classifier.py /path/to/features /path/to/labels.csv --analyze_patterns


# Hybrid
# Train hybrid model only
# python hybrid_attention_classifier.py /path/to/features /path/to/labels.csv
#
# # Compare all approaches
# python hybrid_attention_classifier.py /path/to/features /path/to/labels.csv --compare_all
#
# # Train with custom architecture
# python hybrid_attention_classifier.py /path/to/features /path/to/labels.csv \
#   --projection_dim 256 --n_attention_layers 3 --n_heads 16 --epochs 150
#
# # Visualize attention patterns
# python hybrid_attention_classifier.py /path/to/features /path/to/labels.csv \
#   --visualize_attention



# Full comprehensive analysis
python run_frailty_classification.py \
    --csv_dir /path/to/your/csvs \
    --labels /path/to/your/labels.csv \
    --output_dir results/my_analysis

# On JHPCE with SLURM
sbatch run_jhpce.slurm /path/to/csvs /path/to/labels.csv
