# Activate environment
conda activate ssl-wearables  # or source ssl-wearables/bin/activate for UV

# Generate sample data for testing
python generate_sample_data.py

# Run quick test
python run_frailty_classification.py --csv_dir data/sample_csvs --labels data/sample_labels.csv --quick