"""
Hybrid Attention-Based Classifier: End-to-end deep learning with learnable aggregation
Combines the benefits of both patient-level and window-level approaches
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for window aggregation"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.W_o(attention_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output, attention_weights.mean(dim=1)  # Average attention across heads
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal information"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class WindowEncoder(nn.Module):
    """Encode individual windows with optional fine-tuning of SSL backbone"""
    
    def __init__(self, ssl_backbone, freeze_backbone: bool = False, 
                 projection_dim: int = 512):
        super().__init__()
        
        self.ssl_backbone = ssl_backbone
        
        # Freeze or fine-tune backbone
        if freeze_backbone:
            for param in self.ssl_backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 3000)  # Dummy window
            backbone_output = self.ssl_backbone(dummy_input)
            backbone_dim = backbone_output.shape[-1]
        
        # Projection layer to reduce dimensionality if needed
        if projection_dim != backbone_dim:
            self.projection = nn.Sequential(
                nn.Linear(backbone_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = nn.Identity()
        
        self.output_dim = projection_dim
        
    def forward(self, windows):
        """
        Args:
            windows: (batch_size, n_windows, 3, window_length) or (batch_size * n_windows, 3, window_length)
        Returns:
            encoded_windows: (batch_size, n_windows, projection_dim)
        """
        original_shape = windows.shape
        
        if len(original_shape) == 4:  # (batch_size, n_windows, 3, window_length)
            batch_size, n_windows = original_shape[:2]
            windows = windows.view(-1, *original_shape[2:])  # (batch_size * n_windows, 3, window_length)
        else:  # Already flattened
            batch_size = n_windows = None
        
        # Extract features using SSL backbone
        features = self.ssl_backbone(windows)  # (batch_size * n_windows, backbone_dim)
        
        # Project features
        projected_features = self.projection(features)  # (batch_size * n_windows, projection_dim)
        
        # Reshape back if needed
        if batch_size is not None and n_windows is not None:
            projected_features = projected_features.view(batch_size, n_windows, -1)
        
        return projected_features

class HybridAttentionClassifier(nn.Module):
    """
    Hybrid classifier with attention-based window aggregation
    """
    
    def __init__(self, ssl_backbone, freeze_backbone: bool = False,
                 projection_dim: int = 512, n_attention_layers: int = 2,
                 n_heads: int = 8, dropout: float = 0.1,
                 classifier_hidden_dims: List[int] = None,
                 use_positional_encoding: bool = True):
        super().__init__()
        
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 128]
        
        # Window encoder
        self.window_encoder = WindowEncoder(
            ssl_backbone, freeze_backbone, projection_dim
        )
        
        # Positional encoding for temporal information
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(projection_dim)
        
        # Multi-layer attention
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(projection_dim, n_heads, dropout)
            for _ in range(n_attention_layers)
        ])
        
        # Global pooling strategies
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        classifier_layers = []
        current_dim = projection_dim
        
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Output layer
        classifier_layers.append(nn.Linear(current_dim, 1))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Store attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, windows, mask=None, return_attention=False):
        """
        Forward pass with attention-based aggregation
        
        Args:
            windows: (batch_size, n_windows, 3, window_length) or pre-extracted features
            mask: (batch_size, n_windows) - 1 for valid windows, 0 for padding
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch_size, 1)
            attention_weights: (batch_size, n_windows) if return_attention=True
        """
        batch_size = windows.shape[0]
        
        # Encode windows
        if len(windows.shape) == 4:  # Raw windows
            encoded_windows = self.window_encoder(windows)  # (batch_size, n_windows, projection_dim)
        else:  # Pre-extracted features
            encoded_windows = windows
        
        # Add positional encoding
        if self.use_positional_encoding:
            encoded_windows = self.positional_encoding(encoded_windows)
        
        # Apply attention layers
        attention_outputs = encoded_windows
        all_attention_weights = []
        
        for attention_layer in self.attention_layers:
            attention_outputs, attention_weights = attention_layer(
                attention_outputs, attention_outputs, attention_outputs, mask
            )
            all_attention_weights.append(attention_weights)
        
        # Global aggregation using attention
        if mask is not None:
            # Mask out padded windows
            mask_expanded = mask.unsqueeze(-1).expand_as(attention_outputs)
            attention_outputs = attention_outputs * mask_expanded
        
        # Global average pooling
        aggregated_features = attention_outputs.mean(dim=1)  # (batch_size, projection_dim)
        
        # Classification
        logits = self.classifier(aggregated_features)
        
        # Store attention weights for interpretability
        self.attention_weights = all_attention_weights[-1] if all_attention_weights else None
        
        if return_attention:
            return logits, all_attention_weights
        
        return logits

class PatientDataset(Dataset):
    """Dataset for patient-level data with variable number of windows"""
    
    def __init__(self, patient_features: Dict[str, np.ndarray], 
                 patient_labels: Dict[str, int],
                 max_windows: Optional[int] = None,
                 use_raw_windows: bool = False):
        """
        Args:
            patient_features: Dict {patient_id: features or windows}
            patient_labels: Dict {patient_id: label}
            max_windows: Maximum windows per patient (for batching)
            use_raw_windows: Whether features are raw windows or extracted features
        """
        # Filter to patients with both features and labels
        common_patients = set(patient_features.keys()) & set(patient_labels.keys())
        self.patient_ids = [pid for pid in common_patients if len(patient_features[pid]) > 0]
        
        self.patient_features = {pid: patient_features[pid] for pid in self.patient_ids}
        self.patient_labels = {pid: patient_labels[pid] for pid in self.patient_ids}
        self.use_raw_windows = use_raw_windows
        
        # Determine max windows for padding
        if max_windows is None:
            self.max_windows = max(len(features) for features in self.patient_features.values())
        else:
            self.max_windows = max_windows
        
        # Limit max windows to reasonable number for memory
        self.max_windows = min(self.max_windows, 200)
        
        print(f"Dataset: {len(self.patient_ids)} patients, max {self.max_windows} windows per patient")
        
        # Calculate class weights for balancing
        labels = [self.patient_labels[pid] for pid in self.patient_ids]
        self.class_counts = np.bincount(labels)
        self.class_weights = len(labels) / (2 * self.class_counts)
        
        print(f"Class distribution: {self.class_counts}")
        print(f"Class weights: {self.class_weights}")
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        features = self.patient_features[patient_id]
        label = self.patient_labels[patient_id]
        
        n_windows = len(features)
        
        # Handle different input types
        if self.use_raw_windows:
            # features should be (n_windows, 3, window_length)
            feature_dim = features.shape[1:]
        else:
            # features should be (n_windows, feature_dim)
            feature_dim = (features.shape[1],)
        
        # Pad or truncate to max_windows
        if n_windows > self.max_windows:
            # Randomly sample windows during training, take first during inference
            if self.training if hasattr(self, 'training') else True:
                indices = np.random.choice(n_windows, self.max_windows, replace=False)
                indices.sort()  # Maintain temporal order
            else:
                indices = np.arange(self.max_windows)
            
            features = features[indices]
            mask = torch.ones(self.max_windows)
            
        elif n_windows < self.max_windows:
            # Pad with zeros
            if self.use_raw_windows:
                padding_shape = (self.max_windows - n_windows,) + feature_dim
            else:
                padding_shape = (self.max_windows - n_windows, feature_dim[0])
            
            padding = np.zeros(padding_shape, dtype=features.dtype)
            features = np.vstack([features, padding])
            
            # Create mask
            mask = torch.cat([
                torch.ones(n_windows), 
                torch.zeros(self.max_windows - n_windows)
            ])
        else:
            mask = torch.ones(self.max_windows)
        
        return {
            'features': torch.FloatTensor(features),
            'label': torch.FloatTensor([label]),
            'mask': mask,
            'patient_id': patient_id,
            'n_windows': n_windows,
            'class_weight': self.class_weights[label]
        }

class HybridAttentionTrainer:
    """Trainer for the hybrid attention classifier"""
    
    def __init__(self, ssl_backbone, device: str = None, **model_kwargs):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ssl_backbone = ssl_backbone.to(self.device)
        self.model_kwargs = model_kwargs
        
        # Initialize model
        self.model = HybridAttentionClassifier(
            self.ssl_backbone, **model_kwargs
        ).to(self.device)
        
        print(f"Hybrid attention classifier initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def train(self, patient_features: Dict[str, np.ndarray],
             patient_labels: Dict[str, int],
             test_size: float = 0.2,
             validation_split: float = 0.1,
             epochs: int = 100,
             batch_size: int = 8,  # Smaller batch size due to attention memory requirements
             lr: float = 0.001,
             weight_decay: float = 1e-4,
             early_stopping_patience: int = 15,
             use_class_weights: bool = True,
             **kwargs) -> Dict:
        """
        Train the hybrid attention classifier
        
        Args:
            patient_features: Patient features dictionary
            patient_labels: Patient labels dictionary
            test_size: Test set proportion
            validation_split: Validation set proportion (from training set)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: Weight decay for regularization
            early_stopping_patience: Early stopping patience
            use_class_weights: Whether to use class weights for imbalanced data
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        print("Preparing datasets...")
        
        # Split patients (not windows) to avoid data leakage
        patient_ids = list(set(patient_features.keys()) & set(patient_labels.keys()))
        patient_ids = [pid for pid in patient_ids if len(patient_features[pid]) > 0]
        
        labels_for_split = [patient_labels[pid] for pid in patient_ids]
        
        # Train/test split
        train_patients, test_patients = train_test_split(
            patient_ids, test_size=test_size, random_state=42,
            stratify=labels_for_split
        )
        
        # Train/validation split
        train_labels_for_split = [patient_labels[pid] for pid in train_patients]
        if validation_split > 0:
            train_patients, val_patients = train_test_split(
                train_patients, test_size=validation_split, random_state=42,
                stratify=train_labels_for_split
            )
        else:
            val_patients = []
        
        print(f"Train patients: {len(train_patients)}")
        print(f"Validation patients: {len(val_patients)}")
        print(f"Test patients: {len(test_patients)}")
        
        # Create datasets
        train_features = {pid: patient_features[pid] for pid in train_patients}
        train_labels = {pid: patient_labels[pid] for pid in train_patients}
        
        test_features = {pid: patient_features[pid] for pid in test_patients}
        test_labels = {pid: patient_labels[pid] for pid in test_patients}
        
        train_dataset = PatientDataset(
            train_features, train_labels, 
            max_windows=kwargs.get('max_windows', None),
            use_raw_windows=kwargs.get('use_raw_windows', False)
        )
        
        test_dataset = PatientDataset(
            test_features, test_labels,
            max_windows=train_dataset.max_windows,  # Use same max_windows
            use_raw_windows=kwargs.get('use_raw_windows', False)
        )
        
        if val_patients:
            val_features = {pid: patient_features[pid] for pid in val_patients}
            val_labels = {pid: patient_labels[pid] for pid in val_patients}
            val_dataset = PatientDataset(
                val_features, val_labels,
                max_windows=train_dataset.max_windows,
                use_raw_windows=kwargs.get('use_raw_windows', False)
            )
        else:
            val_dataset = None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=kwargs.get('num_workers', 0)
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=kwargs.get('num_workers', 0)
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=kwargs.get('num_workers', 0)
            )
        else:
            val_loader = None
        
        # Training setup
        if use_class_weights:
            class_weights = torch.FloatTensor(train_dataset.class_weights).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=early_stopping_patience//2, 
            factor=0.5, verbose=True
        )
        
        # Training loop
        train_losses = []
        train_aucs = []
        val_aucs = []
        test_aucs = []
        
        best_val_auc = 0
        best_test_auc = 0
        patience_counter = 0
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            train_preds = []
            train_labels_epoch = []
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in train_pbar:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(features, mask)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Store predictions for AUC calculation
                probs = torch.sigmoid(logits)
                train_preds.extend(probs.detach().cpu().numpy())
                train_labels_epoch.extend(labels.detach().cpu().numpy())
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Calculate training metrics
            train_loss = epoch_loss / len(train_loader)
            train_auc = roc_auc_score(train_labels_epoch, train_preds)
            
            train_losses.append(train_loss)
            train_aucs.append(train_auc)
            
            # Validation phase
            if val_loader:
                val_auc = self._evaluate(val_loader)
                val_aucs.append(val_auc)
                
                # Learning rate scheduling
                scheduler.step(val_auc)
                
                # Early stopping check
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_hybrid_model.pth')
                else:
                    patience_counter += 1
            
            # Test evaluation (for monitoring, not for early stopping)
            test_auc = self._evaluate(test_loader)
            test_aucs.append(test_auc)
            
            if test_auc > best_test_auc:
                best_test_auc = test_auc
            
            # Logging
            log_msg = f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}"
            if val_loader:
                log_msg += f", Val AUC: {val_auc:.4f}"
            log_msg += f", Test AUC: {test_auc:.4f}"
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(log_msg)
            
            # Early stopping
            if val_loader and patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model if validation was used
        if val_loader and os.path.exists('best_hybrid_model.pth'):
            self.model.load_state_dict(torch.load('best_hybrid_model.pth'))
            print("Loaded best model from validation")
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_train_results = self._detailed_evaluate(train_loader, "Training")
        final_test_results = self._detailed_evaluate(test_loader, "Test")
        
        if val_loader:
            final_val_results = self._detailed_evaluate(val_loader, "Validation")
        else:
            final_val_results = None
        
        # Compile results
        results = {
            'train_auc': final_train_results['auc'],
            'test_auc': final_test_results['auc'],
            'val_auc': final_val_results['auc'] if final_val_results else None,
            'best_val_auc': best_val_auc,
            'best_test_auc': best_test_auc,
            'training_history': {
                'train_losses': train_losses,
                'train_aucs': train_aucs,
                'val_aucs': val_aucs,
                'test_aucs': test_aucs
            },
            'train_predictions': final_train_results['predictions'],
            'test_predictions': final_test_results['predictions'],
            'train_labels': final_train_results['labels'],
            'test_labels': final_test_results['labels'],
            'train_patient_ids': final_train_results['patient_ids'],
            'test_patient_ids': final_test_results['patient_ids'],
            'train_attention_weights': final_train_results['attention_weights'],
            'test_attention_weights': final_test_results['attention_weights']
        }
        
        return results
    
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Quick evaluation for monitoring during training"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                logits = self.model(features, mask)
                probs = torch.sigmoid(logits)
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return roc_auc_score(all_labels, all_preds)
    
    def _detailed_evaluate(self, dataloader: DataLoader, split_name: str) -> Dict:
        """Detailed evaluation with attention weights"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_patient_ids = []
        all_attention_weights = []
        
        print(f"Evaluating {split_name} set...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)
                patient_ids = batch['patient_id']
                
                # Get predictions and attention weights
                logits, attention_weights = self.model(features, mask, return_attention=True)
                probs = torch.sigmoid(logits)
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_patient_ids.extend(patient_ids)
                
                # Store attention weights (last layer)
                if attention_weights:
                    all_attention_weights.extend(attention_weights[-1].cpu().numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        
        print(f"{split_name} AUC: {auc:.4f}")
        
        return {
            'auc': auc,
            'predictions': np.array(all_preds).flatten(),
            'labels': np.array(all_labels).flatten(),
            'patient_ids': all_patient_ids,
            'attention_weights': all_attention_weights
        }
    
    def predict(self, patient_features: Dict[str, np.ndarray],
               batch_size: int = 8, return_attention: bool = False) -> Dict:
        """
        Make predictions on new patients
        
        Args:
            patient_features: Patient features dictionary
            batch_size: Batch size for prediction
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions and optionally attention weights
        """
        # Create dummy labels for dataset
        dummy_labels = {pid: 0 for pid in patient_features.keys()}
        
        dataset = PatientDataset(
            patient_features, dummy_labels,
            max_windows=getattr(self, 'max_windows', None),
            use_raw_windows=getattr(self, 'use_raw_windows', False)
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        predictions = {}
        attention_weights_dict = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                features = batch['features'].to(self.device)
                mask = batch['mask'].to(self.device)
                patient_ids = batch['patient_id']
                
                if return_attention:
                    logits, attention_weights = self.model(features, mask, return_attention=True)
                    # Store attention weights
                    for i, pid in enumerate(patient_ids):
                        attention_weights_dict[pid] = attention_weights[-1][i].cpu().numpy()
                else:
                    logits = self.model(features, mask)
                
                probs = torch.sigmoid(logits)
                
                # Store predictions
                for i, pid in enumerate(patient_ids):
                    predictions[pid] = probs[i].item()
        
        result = {'predictions': predictions}
        if return_attention:
            result['attention_weights'] = attention_weights_dict
        
        return result
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_kwargs': self.model_kwargs,
            'ssl_backbone_state_dict': self.ssl_backbone.state_dict()
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        
        # Recreate model
        self.model = HybridAttentionClassifier(
            self.ssl_backbone, **model_data['model_kwargs']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(model_data['model_state_dict'])
        
        print(f"Model loaded from {filepath}")

def visualize_attention_patterns(attention_weights: Dict[str, np.ndarray],
                               patient_labels: Dict[str, int],
                               save_path: str = None):
    """
    Visualize attention patterns across patients
    
    Args:
        attention_weights: Dict {patient_id: attention_weights}
        patient_labels: Dict {patient_id: label}
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Separate by class
    frail_attention = []
    nonfrail_attention = []
    
    for patient_id, weights in attention_weights.items():
        if patient_id in patient_labels:
            if patient_labels[patient_id] == 1:
                frail_attention.append(weights)
            else:
                nonfrail_attention.append(weights)
    
    # 1. Average attention patterns
    if frail_attention and nonfrail_attention:
        # Pad sequences to same length for averaging
        max_len = max(
            max(len(w) for w in frail_attention),
            max(len(w) for w in nonfrail_attention)
        )
        
        frail_padded = []
        nonfrail_padded = []
        
        for weights in frail_attention:
            padded = np.pad(weights, (0, max_len - len(weights)), 'constant', constant_values=0)
            frail_padded.append(padded)
        
        for weights in nonfrail_attention:
            padded = np.pad(weights, (0, max_len - len(weights)), 'constant', constant_values=0)
            nonfrail_padded.append(padded)
        
        frail_mean = np.mean(frail_padded, axis=0)
        nonfrail_mean = np.mean(nonfrail_padded, axis=0)
        
        x = np.arange(len(frail_mean))
        axes[0, 0].plot(x, frail_mean, label='Frail', color='red', linewidth=2)
        axes[0, 0].plot(x, nonfrail_mean, label='Non-frail', color='blue', linewidth=2)
        axes[0, 0].set_xlabel('Window Index')
        axes[0, 0].set_ylabel('Average Attention Weight')
        axes[0, 0].set_title('Average Attention Patterns by Class')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Attention distribution
    all_frail_weights = np.concatenate(frail_attention) if frail_attention else np.array([])
    all_nonfrail_weights = np.concatenate(nonfrail_attention) if nonfrail_attention else np.array([])
    
    if len(all_frail_weights) > 0 and len(all_nonfrail_weights) > 0:
        axes[0, 1].hist(all_nonfrail_weights, alpha=0.7, label='Non-frail', bins=50, density=True)
        axes[0, 1].hist(all_frail_weights, alpha=0.7, label='Frail', bins=50, density=True)
        axes[0, 1].set_xlabel('Attention Weight')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Distribution of Attention Weights')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Attention entropy (measure of focus)
    frail_entropies = []
    nonfrail_entropies = []
    
    for patient_id, weights in attention_weights.items():
        if patient_id in patient_labels and len(weights) > 0:
            # Calculate entropy
            weights_norm = weights / (np.sum(weights) + 1e-8)
            entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
            
            if patient_labels[patient_id] == 1:
                frail_entropies.append(entropy)
            else:
                nonfrail_entropies.append(entropy)
    
    if frail_entropies and nonfrail_entropies:
        axes[0, 2].hist(nonfrail_entropies, alpha=0.7, label='Non-frail', bins=20)
        axes[0, 2].hist(frail_entropies, alpha=0.7, label='Frail', bins=20)
        axes[0, 2].set_xlabel('Attention Entropy')
        axes[0, 2].set_ylabel('Number of Patients')
        axes[0, 2].set_title('Attention Focus (Lower = More Focused)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Heatmap of individual patients
    if frail_attention and nonfrail_attention:
        # Show first few patients from each class
        n_examples = min(5, len(frail_attention), len(nonfrail_attention))
        
        # Prepare data for heatmap
        heatmap_data = []
        labels_for_heatmap = []
        
        for i in range(n_examples):
            # Pad to same length
            max_len_example = max(len(frail_attention[i]), len(nonfrail_attention[i]))
            
            frail_padded = np.pad(frail_attention[i], (0, max_len_example - len(frail_attention[i])))
            nonfrail_padded = np.pad(nonfrail_attention[i], (0, max_len_example - len(nonfrail_attention[i])))
            
            heatmap_data.append(frail_padded)
            heatmap_data.append(nonfrail_padded)
            labels_for_heatmap.extend([f'Frail {i+1}', f'Non-frail {i+1}'])
        
        if heatmap_data:
            heatmap_array = np.array(heatmap_data)
            sns.heatmap(heatmap_array, ax=axes[1, 0], cmap='Blues', 
                       yticklabels=labels_for_heatmap, cbar=True)
            axes[1, 0].set_xlabel('Window Index')
            axes[1, 0].set_title('Individual Patient Attention Patterns')
    
    # 5. Peak attention analysis
    frail_peaks = []
    nonfrail_peaks = []
    
    for patient_id, weights in attention_weights.items():
        if patient_id in patient_labels and len(weights) > 0:
            # Find peak attention position (normalized)
            peak_pos = np.argmax(weights) / len(weights)
            
            if patient_labels[patient_id] == 1:
                frail_peaks.append(peak_pos)
            else:
                nonfrail_peaks.append(peak_pos)
    
    if frail_peaks and nonfrail_peaks:
        axes[1, 1].hist(nonfrail_peaks, alpha=0.7, label='Non-frail', bins=20)
        axes[1, 1].hist(frail_peaks, alpha=0.7, label='Frail', bins=20)
        axes[1, 1].set_xlabel('Peak Attention Position (Normalized)')
        axes[1, 1].set_ylabel('Number of Patients')
        axes[1, 1].set_title('Where Does the Model Focus Most?')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Attention vs prediction confidence
    if frail_attention and nonfrail_attention:
        # Calculate max attention weight as a measure of confidence
        frail_max_attention = [np.max(weights) for weights in frail_attention]
        nonfrail_max_attention = [np.max(weights) for weights in nonfrail_attention]
        
        # Box plot
        data_for_box = [nonfrail_max_attention, frail_max_attention]
        axes[1, 2].boxplot(data_for_box, labels=['Non-frail', 'Frail'])
        axes[1, 2].set_ylabel('Max Attention Weight')
        axes[1, 2].set_title('Attention Confidence by Class')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    plt.show()

def compare_all_approaches(patient_features: Dict[str, np.ndarray],
                         patient_labels: Dict[str, int],
                         ssl_backbone,
                         output_dir: str = "comparison_results") -> pd.DataFrame:
    """
    Compare all three approaches: Patient-level, Window-level, and Hybrid
    
    Args:
        patient_features: Patient features dictionary
        patient_labels: Patient labels dictionary
        ssl_backbone: SSL backbone model
        output_dir: Directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print("=== COMPREHENSIVE APPROACH COMPARISON ===\n")
    
    # 1. Patient-Level Classifier
    print("1. Testing Patient-Level Classifier...")
    try:
        from patient_level_classifier import PatientLevelClassifier
        
        patient_classifier = PatientLevelClassifier(classifier_type='random_forest')
        patient_results = patient_classifier.train(
            patient_features, patient_labels,
            aggregation_method='comprehensive',
            optimize_hyperparameters=False
        )
        
        results.append({
            'approach': 'Patient-Level',
            'method': 'Random Forest + Comprehensive Aggregation',
            'auc': patient_results['test_auc'],
            'cv_auc': patient_results['cv_auc_mean'],
            'cv_std': patient_results['cv_auc_std']
        })
        
        print(f"   Patient-Level AUC: {patient_results['test_auc']:.4f}")
        
    except Exception as e:
        print(f"   Error with Patient-Level: {e}")
    
    # 2. Window-Level Classifier
    print("\n2. Testing Window-Level Classifier...")
    try:
        from window_level_classifier import WindowLevelClassifier
        
        window_classifier = WindowLevelClassifier(classifier_type='random_forest')
        window_results = window_classifier.train(
            patient_features, patient_labels,
            label_strategy='inherit',
            validation_strategy='patient_split'
        )
        
        # Get best patient-level aggregation
        patient_eval = window_classifier.evaluate_patient_level(window_results, patient_labels)
        best_patient_auc = patient_eval['patient_auc'].max()
        best_method = patient_eval.iloc[0]['aggregation_method']
        
        results.append({
            'approach': 'Window-Level',
            'method': f'Random Forest + {best_method}',
            'auc': best_patient_auc,
            'window_auc': window_results['window_test_auc'],
            'cv_auc': None,
            'cv_std': None
        })
        
        print(f"   Window-Level AUC: {window_results['window_test_auc']:.4f}")
        print(f"   Best Patient AUC: {best_patient_auc:.4f} ({best_method})")
        
    except Exception as e:
        print(f"   Error with Window-Level: {e}")
    
    # 3. Hybrid Attention Classifier
    print("\n3. Testing Hybrid Attention Classifier...")
    try:
        hybrid_trainer = HybridAttentionTrainer(
            ssl_backbone,
            freeze_backbone=False,
            projection_dim=256,
            n_attention_layers=2,
            n_heads=8
        )
        
        hybrid_results = hybrid_trainer.train(
            patient_features, patient_labels,
            epochs=50,  # Reduced for comparison
            batch_size=4,  # Small batch for memory
            early_stopping_patience=10
        )
        
        results.append({
            'approach': 'Hybrid Attention',
            'method': 'End-to-end + Multi-head Attention',
            'auc': hybrid_results['test_auc'],
            'best_auc': hybrid_results['best_test_auc'],
            'cv_auc': None,
            'cv_std': None
        })
        
        print(f"   Hybrid Attention AUC: {hybrid_results['test_auc']:.4f}")
        print(f"   Best Test AUC: {hybrid_results['best_test_auc']:.4f}")
        
        # Save attention visualization
        if hybrid_results['test_attention_weights']:
            attention_dict = {
                pid: weights for pid, weights in 
                zip(hybrid_results['test_patient_ids'], hybrid_results['test_attention_weights'])
            }
            
            visualize_attention_patterns(
                attention_dict, patient_labels,
                os.path.join(output_dir, "attention_patterns.png")
            )
        
    except Exception as e:
        print(f"   Error with Hybrid Attention: {e}")
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        df_results = df_results.sort_values('auc', ascending=False)
        
        print(f"\n=== FINAL COMPARISON RESULTS ===")
        print(df_results.to_string(index=False))
        
        # Save results
        df_results.to_csv(os.path.join(output_dir, "approach_comparison.csv"), index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Main AUC comparison
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(df_results)), df_results['auc'], 
                      color=['skyblue', 'lightcoral', 'lightgreen'][:len(df_results)])
        plt.xlabel('Approach')
        plt.ylabel('Test AUC')
        plt.title('Approach Comparison - Test AUC')
        plt.xticks(range(len(df_results)), df_results['approach'], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Cross-validation comparison (if available)
        plt.subplot(2, 2, 2)
        cv_data = df_results[df_results['cv_auc'].notna()]
        if len(cv_data) > 0:
            plt.errorbar(range(len(cv_data)), cv_data['cv_auc'], 
                        yerr=cv_data['cv_std'], fmt='o-', capsize=5)
            plt.xlabel('Approach')
            plt.ylabel('CV AUC')
            plt.title('Cross-Validation Results')
            plt.xticks(range(len(cv_data)), cv_data['approach'], rotation=45)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No CV results available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        # Method details
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        table_data = []
        for _, row in df_results.iterrows():
            table_data.append([
                row['approach'],
                row['method'],
                f"{row['auc']:.4f}",
                f"{row.get('cv_auc', 'N/A'):.4f}" if pd.notna(row.get('cv_auc')) else 'N/A'
            ])
        
        table = plt.table(cellText=table_data,
                         colLabels=['Approach', 'Method', 'Test AUC', 'CV AUC'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_plot.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Winner announcement
        winner = df_results.iloc[0]
        print(f"\n🏆 WINNER: {winner['approach']} ({winner['method']})")
        print(f"   Best AUC: {winner['auc']:.4f}")
        
    else:
        print("No successful results to compare!")
    
    return df_results

def load_patient_features(features_dir: str) -> Dict[str, np.ndarray]:
    """Load patient features from directory of pickle files"""
    import glob
    
    feature_files = glob.glob(os.path.join(features_dir, "*_features.pkl"))
    patient_features = {}
    
    print(f"Loading features from {len(feature_files)} files...")
    
    for file_path in tqdm(feature_files):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            patient_id = data['patient_id']
            features = data['features']
            
            if len(features) > 0:
                patient_features[patient_id] = features
            else:
                print(f"Warning: No features for {patient_id}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded features for {len(patient_features)} patients")
    return patient_features

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train hybrid attention classifier")
    parser.add_argument("features_dir", help="Directory containing extracted features")
    parser.add_argument("labels_file", help="CSV file with patient labels")
    parser.add_argument("--output_dir", "-o", default="hybrid_results", 
                       help="Output directory")
    parser.add_argument("--ssl_model_path", help="Path to SSL backbone model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--projection_dim", type=int, default=512, 
                       help="Projection dimension")
    parser.add_argument("--n_attention_layers", type=int, default=2,
                       help="Number of attention layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="Freeze SSL backbone weights")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all approaches")
    parser.add_argument("--visualize_attention", action="store_true",
                       help="Create attention visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    patient_features = load_patient_features(args.features_dir)
    
    # Load labels
    labels_df = pd.read_csv(args.labels_file)
    patient_labels = dict(zip(labels_df['patient_id'], labels_df['label']))
    
    print(f"Loaded {len(patient_features)} patients with features")
    print(f"Loaded {len(patient_labels)} patients with labels")
    
    # Find intersection
    common_patients = set(patient_features.keys()) & set(patient_labels.keys())
    print(f"Common patients: {len(common_patients)}")
    
    if len(common_patients) == 0:
        print("Error: No common patients between features and labels!")
        exit(1)
    
    # Filter to common patients
    patient_features = {pid: patient_features[pid] for pid in common_patients}
    patient_labels = {pid: patient_labels[pid] for pid in common_patients}
    
    # Load SSL backbone
    if args.ssl_model_path and os.path.exists(args.ssl_model_path):
        ssl_backbone = torch.load(args.ssl_model_path)
    else:
        # Use placeholder model
        from feature_extraction import SSLWearablesBackbone
        ssl_backbone = SSLWearablesBackbone(n_features=1024)
    
    if args.compare_all:
        # Compare all approaches
        comparison_results = compare_all_approaches(
            patient_features, patient_labels, ssl_backbone, args.output_dir
        )
    else:
        # Train only hybrid model
        print("Training Hybrid Attention Classifier...")
        
        trainer = HybridAttentionTrainer(
            ssl_backbone,
            freeze_backbone=args.freeze_backbone,
            projection_dim=args.projection_dim,
            n_attention_layers=args.n_attention_layers,
            n_heads=args.n_heads
        )
        
        results = trainer.train(
            patient_features, patient_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        # Save model
        model_path = os.path.join(args.output_dir, "hybrid_attention_model.pth")
        trainer.save_model(model_path)
        
        # Save results
        results_path = os.path.join(args.output_dir, "training_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Visualize attention if requested
        if args.visualize_attention and results['test_attention_weights']:
            attention_dict = {
                pid: weights for pid, weights in 
                zip(results['test_patient_ids'], results['test_attention_weights'])
            }
            
            visualize_attention_patterns(
                attention_dict, patient_labels,
                os.path.join(args.output_dir, "attention_patterns.png")
            )
        
        print(f"\nTraining complete! Results saved to {args.output_dir}")
        print(f"Final Test AUC: {results['test_auc']:.4f}")
        print(f"Best Test AUC: {results['best_test_auc']:.4f}")            