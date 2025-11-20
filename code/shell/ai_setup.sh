#!/bin/bash
# Complete setup script for the SSL-Wearables Frailty Classification Pipeline

set -e

echo "🔧 Setting up SSL-Wearables Frailty Classification Pipeline"

# Check if we're on JHPCE
if [[ $HOSTNAME == *"jhpce"* ]]; then
    echo "✅ Detected JHPCE environment"
    SETUP_METHOD="jhpce"
elif command -v uv &> /dev/null; then
    echo "✅ UV detected - using fast setup"
    SETUP_METHOD="uv"
elif command -v conda &> /dev/null; then
    echo "✅ Conda detected - using conda setup"
    SETUP_METHOD="conda"
else
    echo "❌ Neither conda nor uv found. Please install one of them first."
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p {data,models,results,configs,logs}

# Setup environment based on method
case $SETUP_METHOD in
    "jhpce")
        echo "🏥 Setting up for JHPCE..."
        # Load JHPCE modules
        module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || echo "⚠️ No CUDA module loaded"
        module load python/3.9 2>/dev/null || echo "Using system Python"
        
        # Create conda environment
        conda env create -f environment.yml -n ssl-wearables 2>/dev/null || conda create -n ssl-wearables python=3.9 -y
        
        # Activate and install packages
        source activate ssl-wearables
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y 2>/dev/null || \
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        ;;
        
    "uv")
        echo "⚡ Setting up with UV (fast)..."
        uv venv ssl-wearables --python 3.9
        source ssl-wearables/bin/activate
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        uv pip install -r requirements.txt
        ;;
        
    "conda")
        echo "🐍 Setting up with Conda..."
        conda env create -f environment.yml 2>/dev/null || {
            conda create -n ssl-wearables python=3.9 -y
            conda activate ssl-wearables
            conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        }
        ;;
esac

# Install additional packages
echo "📦 Installing additional packages..."
pip install accelerometer-features neurokit2 imbalanced-learn optuna shap wandb tqdm

# Test installation
echo "🧪 Testing installation..."
python -c "
import torch
import numpy as np
import pandas as pd
import sklearn
print('✅ All packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
"

# Create sample configuration files
echo "📝 Creating sample configuration files..."

cat > configs/quick_test.json << 'EOF'
{
  "data_type": "csv_directory",
  "csv_directory": "data/sample_csvs",
  "labels_file": "data/sample_labels.csv",
  "output_dir": "results/quick_test",
  "run_patient_level": true,
  "run_window_level": false,
  "run_hybrid": false,
  "compare_approaches": false,
  "feature_extraction": {
    "window_size": 1500,
    "overlap": 0.5,
    "target_hz": 50,
    "batch_size": 32
  },
  "patient_level": {
    "classifier_type": "random_forest",
    "aggregation_method": "basic",
    "compare_aggregations": false,
    "optimize_hyperparameters": false,
    "test_size": 0.3
  }
}
EOF

cat > configs/comprehensive.json << 'EOF'
{
  "data_type": "csv_directory",
  "output_dir": "results/comprehensive",
  "run_patient_level": true,
  "run_window_level": true,
  "run_hybrid": true,
  "compare_approaches": true,
  "feature_extraction": {
    "window_size": 3000,
    "overlap": 0.5,
    "target_hz": 100,
    "batch_size": 64
  },
  "patient_level": {
    "classifier_type": "random_forest",
    "aggregation_method": "comprehensive",
    "compare_aggregations": true,
    "optimize_hyperparameters": true
  },
  "window_level": {
    "classifier_type": "random_forest",
    "compare_strategies": true,
    "compare_classifiers": true
  },
  "hybrid_attention": {
    "epochs": 100,
    "batch_size": 8,
    "visualize_attention": true
  }
}
EOF

# Create sample SLURM script for JHPCE
if [[ $SETUP_METHOD == "jhpce" ]]; then
cat > run_jhpce.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=frailty_classification
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/frailty_%j.out
#SBATCH --error=logs/frailty_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load cuda/12.1
module load python/3.9

# Activate environment
source activate ssl-wearables

# Test CUDA
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run pipeline
python run_frailty_classification.py \
    --csv_dir $1 \
    --labels $2 \
    --output_dir results/jhpce_run_$(date +%Y%m%d_%H%M%S)

echo "Job completed at: $(date)"
EOF
chmod +x run_jhpce.slurm
fi

# Create usage examples
cat > examples.sh << 'EOF'
#!/bin/bash
# Usage examples for the Frailty Classification Pipeline

echo "🚀 SSL-Wearables Frailty Classification Pipeline Examples"

# Example 1: Quick test
echo "1. Quick test (patient-level only):"
echo "python run_frailty_classification.py --csv_dir data/csvs --labels data/labels.csv --quick"

# Example 2: Comprehensive analysis
echo "2. Comprehensive analysis:"
echo "python run_frailty_classification.py --config configs/comprehensive.json --csv_dir data/csvs --labels data/labels.csv"

# Example 3: From pre-extracted features
echo "3. From pre-extracted features:"
echo "python run_frailty_classification.py --features_dir data/features --labels data/labels.csv"

# Example 4: Only hybrid attention
echo "4. Only hybrid attention:"
echo "python run_frailty_classification.py --csv_dir data/csvs --labels data/labels.csv --no_patient_level --no_window_level"

# Example 5: JHPCE submission
echo "5. JHPCE submission:"
echo "sbatch run_jhpce.slurm data/csvs data/labels.csv"

# Example 6: Validation only
echo "6. Validate inputs only:"
echo "python run_frailty_classification.py --csv_dir data/csvs --labels data/labels.csv --validate_only"
EOF
chmod +x examples.sh

# Create sample data generator
cat > generate_sample_data.py << 'EOF'
"""
Generate sample data for testing the pipeline
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_sample_csvs(output_dir="data/sample_csvs", n_patients=10):
    """Generate sample CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    patient_labels = {}
    
    for i in range(n_patients):
        patient_id = f"patient_{i:03d}"
        
        # Generate 2 hours of data at 100Hz
        n_samples = 2 * 60 * 60 * 100
        
        # Create timestamps
        start_time = datetime.now() - timedelta(days=i)
        timestamps = [start_time + timedelta(seconds=j/100) for j in range(n_samples)]
        
        # Generate accelerometry data
        if i < n_patients // 2:  # Non-frail
            # More active, higher variability
            x = np.random.normal(0, 1.5, n_samples)
            y = np.random.normal(0, 1.5, n_samples)
            z = np.random.normal(9.81, 1.5, n_samples)  # Include gravity
            
            # Add activity bursts
            for _ in range(20):
                start_idx = np.random.randint(0, n_samples - 1000)
                burst_length = np.random.randint(500, 1000)
                x[start_idx:start_idx + burst_length] += np.random.normal(0, 3, burst_length)
                y[start_idx:start_idx + burst_length] += np.random.normal(0, 3, burst_length)
                z[start_idx:start_idx + burst_length] += np.random.normal(0, 2, burst_length)
            
            patient_labels[patient_id] = 0  # Non-frail
        else:  # Frail
            # Less active, lower variability
            x = np.random.normal(0, 0.8, n_samples)
            y = np.random.normal(0, 0.8, n_samples)
            z = np.random.normal(9.81, 0.8, n_samples)
            
            patient_labels[patient_id] = 1  # Frail
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': timestamps,
            'x': x,
            'y': y,
            'z': z
        })
        
        # Save CSV
        csv_path = os.path.join(output_dir, f"{patient_id}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Generated {csv_path}")
    
    # Save labels
    labels_df = pd.DataFrame([
        {'patient_id': pid, 'label': label}
        for pid, label in patient_labels.items()
    ])
    
    labels_path = "data/sample_labels.csv"
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    labels_df.to_csv(labels_path, index=False)
    print(f"Generated {labels_path}")
    
    print(f"Sample data generation complete!")
    print(f"  Patients: {n_patients}")
    print(f"  Non-frail: {sum(1 for l in patient_labels.values() if l == 0)}")
    print(f"  Frail: {sum(1 for l in patient_labels.values() if l == 1)}")

if __name__ == "__main__":
    generate_sample_csvs()
EOF

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment:"
if [[ $SETUP_METHOD == "uv" ]]; then
    echo "   source ssl-wearables/bin/activate"
else
    echo "   conda activate ssl-wearables"
fi
echo ""
echo "2. Generate sample data (optional):"
echo "   python generate_sample_data.py"
echo ""
echo "3. Run quick test:"
echo "   python run_frailty_classification.py --config configs/quick_test.json"
echo ""
echo "4. See more examples:"
echo "   ./examples.sh"
echo ""
echo "📁 Directory structure created:"
echo "   data/          - Put your CSV files and labels here"
echo "   models/        - Pre-trained models go here"
echo "   results/       - Pipeline outputs will be saved here"
echo "   configs/       - Configuration files"
echo "   logs/          - Log files"
echo ""
echo "🎉 Ready to classify frailty!"