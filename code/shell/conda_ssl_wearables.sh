#!/bin/bash
# Create environment
# From claude: https://chat.ai.jh.edu/c/a990144b-31c6-42f4-a3ad-97e0374361d4
conda env create -f code/shell/environment.yml
conda activate ssl-wearables

# Test installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name()}')
"

echo "✅ Conda environment setup complete!"
echo "Activate with: conda activate ssl-wearables"

