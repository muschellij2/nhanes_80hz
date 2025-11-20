"""
SSL-Wearables Feature Extraction Module
Handles CSV loading, preprocessing, and feature extraction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import pickle
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SSLWearablesBackbone(nn.Module):
    """
    SSL-Wearables backbone model
    Replace this with actual SSL-wearables model loading
    """
    def __init__(self, n_features=1024):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, n_features)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 3, sequence_length)
        features = self.conv_layers(x)
        features = features.squeeze(-1)
        features = self.fc_layers(features)
        return features

class AccelerometryProcessor:
    """Handles CSV loading and preprocessing"""
    
    def __init__(self, target_hz: int = 100):
        self.target_hz = target_hz
    
    def load_csv(self, csv_path: str, time_col: str = 'time', 
                 x_col: str = 'x', y_col: str = 'y', z_col: str = 'z',
                 patient_id: Optional[str] = None) -> Dict:
        """
        Load accelerometry data from CSV
        
        Args:
            csv_path: Path to CSV file
            time_col: Name of timestamp column
            x_col, y_col, z_col: Names of accelerometry columns
            patient_id: Patient identifier
            
        Returns:
            Dictionary with processed data and metadata
        """
        print(f"Loading {csv_path}...")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        # First, lowercase all column names
        df.columns = df.columns.str.lower()
        
        # Map common timestamp column variations to 'time'
        timestamp_mappings = {
            'header_timestamp': 'time',
            'header_time_stamp': 'time',
            'timestamp': 'time',
            'time_stamp': 'time'
        }
        
        # Apply timestamp mappings
        for old_name, new_name in timestamp_mappings.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                print(f"  Renamed column '{old_name}' to '{new_name}'")
                break

        # Check if we have the required columns after normalization
        required_cols = [time_col, x_col, y_col, z_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  Available columns after normalization: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"  Using columns: time='{time_col}', x='{x_col}', y='{y_col}', z='{z_col}'")

        # Parse timestamps
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # Extract accelerometry data
        accel_data = df[[x_col, y_col, z_col]].values
        timestamps = df[time_col].values
        
        # Calculate original sampling rate
        time_diffs = np.diff(timestamps).astype('timedelta64[ms]').astype(float) / 1000.0
        original_hz = 1.0 / np.median(time_diffs)
        
        print(f"  Original sampling rate: {original_hz:.2f} Hz")
        print(f"  Duration: {timestamps[-1] - timestamps[0]}")
        print(f"  Total samples: {len(accel_data)}")
        
        # Resample if needed
        if abs(original_hz - self.target_hz) > 1:
            print(f"  Resampling to {self.target_hz} Hz...")
            accel_data, timestamps = self._resample_data(
                accel_data, timestamps, original_hz, self.target_hz
            )
        
        # Quality assessment
        quality_metrics = self._assess_quality(accel_data)
        
        return {
            'patient_id': patient_id or os.path.basename(csv_path).split('.')[0],
            'data': accel_data,
            'timestamps': timestamps,
            'sampling_rate': self.target_hz,
            'original_hz': original_hz,
            'quality_metrics': quality_metrics
        }
    
    def _resample_data(self, data: np.ndarray, timestamps: np.ndarray,
                      original_hz: float, target_hz: float) -> Tuple[np.ndarray, np.ndarray]:
        """Resample data to target frequency"""
        from scipy.interpolate import interp1d
        
        # Convert to seconds
        time_seconds = (timestamps - timestamps[0]).astype('timedelta64[ms]').astype(float) / 1000.0
        
        # Create new time vector
        duration = time_seconds[-1]
        new_time_seconds = np.arange(0, duration, 1.0/target_hz)
        
        # Interpolate each axis
        resampled_data = np.zeros((len(new_time_seconds), 3))
        for axis in range(3):
            f = interp1d(time_seconds, data[:, axis], kind='linear',
                        bounds_error=False, fill_value='extrapolate')
            resampled_data[:, axis] = f(new_time_seconds)
        
        # Create new timestamps
        new_timestamps = timestamps[0] + pd.to_timedelta(new_time_seconds, unit='s')
        
        return resampled_data, new_timestamps.values
    
    def _assess_quality(self, data: np.ndarray) -> Dict:
        """Assess data quality"""
        return {
            'missing_ratio': np.isnan(data).sum() / data.size,
            'stuck_ratio': np.mean([np.mean(np.diff(data[:, i]) == 0) for i in range(3)]),
            'outlier_ratio': np.mean(np.abs(data) > 8),  # Assuming g units
            'signal_magnitude_area': np.mean(np.sum(np.abs(data), axis=1))
        }

class FeatureExtractor:
    """Main feature extraction class"""
    
    def __init__(self, model_path: Optional[str] = None, batch_size: int = 64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Load SSL-wearables model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = torch.load(model_path, map_location=self.device)
        else:
            print("Using placeholder SSL-wearables model")
            self.model = SSLWearablesBackbone(n_features=1024)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Feature extractor initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def create_windows(self, data: np.ndarray, window_size: int = 3000, 
                      overlap: float = 0.5) -> np.ndarray:
        """
        Create overlapping windows from continuous data
        
        Args:
            data: (n_samples, 3) accelerometry data
            window_size: Window size in samples (default: 30s at 100Hz)
            overlap: Overlap fraction (0-1)
            
        Returns:
            windows: (n_windows, 3, window_size) array
        """
        step_size = int(window_size * (1 - overlap))
        windows = []
        
        for start in range(0, len(data) - window_size + 1, step_size):
            window = data[start:start + window_size].T  # Shape: (3, window_size)
            windows.append(window)
        
        return np.array(windows) if windows else np.empty((0, 3, window_size))
    
    def extract_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract features from windows using SSL-wearables model
        
        Args:
            windows: (n_windows, 3, window_size) array
            
        Returns:
            features: (n_windows, 1024) array
        """
        if len(windows) == 0:
            return np.empty((0, 1024))
        
        # Create DataLoader for efficient batching
        dataset = TensorDataset(torch.FloatTensor(windows))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                batch_data = batch[0].to(self.device)
                features = self.model(batch_data)
                all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def process_patient(self, data_dict: Dict, window_size: int = 3000, 
                       overlap: float = 0.5) -> Dict:
        """
        Complete processing pipeline for one patient
        
        Args:
            data_dict: Output from AccelerometryProcessor.load_csv()
            window_size: Window size in samples
            overlap: Window overlap fraction
            
        Returns:
            Dictionary with extracted features and metadata
        """
        patient_id = data_dict['patient_id']
        accel_data = data_dict['data']
        
        print(f"Processing patient {patient_id}...")
        
        # Create windows
        windows = self.create_windows(accel_data, window_size, overlap)
        print(f"  Created {len(windows)} windows")
        
        if len(windows) == 0:
            print(f"  Warning: No windows created for {patient_id}")
            return {
                'patient_id': patient_id,
                'features': np.empty((0, 1024)),
                'n_windows': 0,
                'quality_metrics': data_dict['quality_metrics'],
                'timestamps': data_dict['timestamps']
            }
        
        # Extract features
        features = self.extract_features(windows)
        print(f"  Extracted features: {features.shape}")
        
        return {
            'patient_id': patient_id,
            'features': features,
            'n_windows': len(windows),
            'quality_metrics': data_dict['quality_metrics'],
            'timestamps': data_dict['timestamps']
        }

def extract_features_from_csv(csv_path: str, output_path: str = None, 
                             patient_id: str = None, **kwargs) -> Dict:
    """
    Complete pipeline from CSV to features
    
    Args:
        csv_path: Path to CSV file
        output_path: Path to save features (optional)
        patient_id: Patient identifier
        **kwargs: Additional arguments for processing
        
    Returns:
        Dictionary with extracted features
    """
    # Initialize components
    processor = AccelerometryProcessor(target_hz=kwargs.get('target_hz', 100))
    extractor = FeatureExtractor(
        model_path=kwargs.get('model_path'),
        batch_size=kwargs.get('batch_size', 64)
    )
    
    # Load and preprocess CSV
    data_dict = processor.load_csv(csv_path, patient_id=patient_id)
    
    # Check data quality
    quality_score = 1 - np.mean(list(data_dict['quality_metrics'].values()))
    if quality_score < 0.7:
        print(f"Warning: Low data quality score: {quality_score:.3f}")
    
    # Extract features
    result = extractor.process_patient(
        data_dict,
        window_size=kwargs.get('window_size', 3000),
        overlap=kwargs.get('overlap', 0.5)
    )
    
    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Features saved to {output_path}")
    
    return result

def batch_extract_features(csv_dir: str, output_dir: str, 
                          pattern: str = "*.csv", **kwargs):
    """
    Extract features from multiple CSV files
    
    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory to save extracted features
        pattern: File pattern to match
        **kwargs: Additional processing arguments
    """
    import glob
    
    csv_files = glob.glob(os.path.join(csv_dir, pattern))
    print(f"Found {len(csv_files)} CSV files")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for csv_path in tqdm(csv_files, desc="Processing files"):
        try:
            patient_id = os.path.basename(csv_path).split('.')[0]
            output_path = os.path.join(output_dir, f"{patient_id}_features.pkl")
            
            result = extract_features_from_csv(
                csv_path, output_path, patient_id, **kwargs
            )
            
            results[patient_id] = result
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            continue
    
    # Save summary
    summary_path = os.path.join(output_dir, "extraction_summary.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Batch extraction complete. Summary saved to {summary_path}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from accelerometry CSV")
    parser.add_argument("csv_path", help="Path to CSV file or directory")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--patient_id", help="Patient identifier")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")
    parser.add_argument("--window_size", type=int, default=3000, help="Window size")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap")
    parser.add_argument("--target_hz", type=int, default=100, help="Target sampling rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_extract_features(
            args.csv_path,
            args.output or "extracted_features",
            window_size=args.window_size,
            overlap=args.overlap,
            target_hz=args.target_hz,
            batch_size=args.batch_size
        )
    else:
        extract_features_from_csv(
            args.csv_path,
            args.output,
            args.patient_id,
            window_size=args.window_size,
            overlap=args.overlap,
            target_hz=args.target_hz,
            batch_size=args.batch_size
        )
