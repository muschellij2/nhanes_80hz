"""
Window-Level Classifier: Classify each window then aggregate predictions
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class WindowDataset(Dataset):
    """Dataset for window-level classification"""
    
    def __init__(self, windows: np.ndarray, labels: np.ndarray, patient_ids: List[str]):
        """
        Args:
            windows: (n_windows, n_features) array of window features
            labels: (n_windows,) array of window labels
            patient_ids: List of patient IDs for each window
        """
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.FloatTensor(labels)
        self.patient_ids = patient_ids
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'features': self.windows[idx],
            'label': self.labels[idx],
            'patient_id': self.patient_ids[idx]
        }

class DeepWindowClassifier(nn.Module):
    """Deep learning classifier for individual windows"""
    
    def __init__(self, input_dim: int = 1024, hidden_dims: List[int] = None, 
                 dropout_rate: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)

class WindowLevelClassifier:
    """Window-level classifier with multiple aggregation strategies"""
    
    def __init__(self, classifier_type: str = 'random_forest', **kwargs):
        """
        Args:
            classifier_type: 'random_forest', 'logistic_regression', or 'deep'
            **kwargs: Additional arguments for classifier initialization
        """
        self.classifier_type = classifier_type
        self.classifier = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kwargs = kwargs
        
        # For tracking patient-window mapping
        self.patient_window_mapping = {}
        
    def _create_window_labels(self, patient_features: Dict[str, np.ndarray],
                             patient_labels: Dict[str, int],
                             label_strategy: str = 'inherit') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create window-level labels from patient-level labels
        
        Args:
            patient_features: Dict {patient_id: features(n_windows, n_features)}
            patient_labels: Dict {patient_id: label}
            label_strategy: Strategy for assigning window labels
                - 'inherit': All windows inherit patient label
                - 'noisy': Add noise to patient labels for windows
                - 'activity_based': Use activity level to modify labels
        
        Returns:
            window_features: (n_windows, n_features)
            window_labels: (n_windows,)
            window_patient_ids: List of patient IDs for each window
        """
        window_features = []
        window_labels = []
        window_patient_ids = []
        
        print(f"Creating window labels using strategy: {label_strategy}")
        
        for patient_id, features in tqdm(patient_features.items(), desc="Processing patients"):
            if patient_id not in patient_labels or len(features) == 0:
                continue
            
            patient_label = patient_labels[patient_id]
            n_windows = len(features)
            
            # Store patient-window mapping
            start_idx = len(window_features)
            self.patient_window_mapping[patient_id] = {
                'start_idx': start_idx,
                'n_windows': n_windows,
                'patient_label': patient_label
            }
            
            if label_strategy == 'inherit':
                # All windows inherit patient label
                labels = [patient_label] * n_windows
                
            elif label_strategy == 'noisy':
                # Add noise to patient labels
                base_prob = 0.9 if patient_label == 1 else 0.1
                labels = np.random.binomial(1, base_prob, n_windows)
                
            elif label_strategy == 'activity_based':
                # Use activity level (feature magnitude) to modify labels
                activity_levels = np.linalg.norm(features, axis=1)
                activity_threshold = np.percentile(activity_levels, 50)
                
                if patient_label == 1:  # Frail patient
                    # Lower activity windows more likely to be frail
                    probs = np.where(activity_levels < activity_threshold, 0.8, 0.4)
                else:  # Non-frail patient
                    # Higher activity windows less likely to be frail
                    probs = np.where(activity_levels > activity_threshold, 0.1, 0.3)
                
                labels = [np.random.binomial(1, p) for p in probs]
            
            else:
                raise ValueError(f"Unknown label strategy: {label_strategy}")
            
            # Add to lists
            window_features.extend(features)
            window_labels.extend(labels)
            window_patient_ids.extend([patient_id] * n_windows)
        
        window_features = np.array(window_features)
        window_labels = np.array(window_labels)
        
        print(f"Created {len(window_features)} windows from {len(patient_features)} patients")
        print(f"Window label distribution: {np.bincount(window_labels)}")
        
        return window_features, window_labels, window_patient_ids
    
    def _initialize_classifier(self):
        """Initialize the window classifier"""
        if self.classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', 10),
                min_samples_split=self.kwargs.get('min_samples_split', 5),
                min_samples_leaf=self.kwargs.get('min_samples_leaf', 2),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        elif self.classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=self.kwargs.get('C', 1.0)
            )
        
        elif self.classifier_type == 'deep':
            input_dim = self.kwargs.get('input_dim', 1024)
            hidden_dims = self.kwargs.get('hidden_dims', [512, 256, 128])
            dropout_rate = self.kwargs.get('dropout_rate', 0.3)
            
            self.classifier = DeepWindowClassifier(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def train(self, patient_features: Dict[str, np.ndarray],
             patient_labels: Dict[str, int],
             label_strategy: str = 'inherit',
             test_size: float = 0.2,
             validation_strategy: str = 'patient_split',
             **train_kwargs) -> Dict:
        """
        Train the window-level classifier
        
        Args:
            patient_features: Patient features dictionary
            patient_labels: Patient labels dictionary
            label_strategy: Strategy for creating window labels
            test_size: Test set proportion
            validation_strategy: 'random_split' or 'patient_split'
            **train_kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        # Create window-level dataset
        X_windows, y_windows, window_patient_ids = self._create_window_labels(
            patient_features, patient_labels, label_strategy
        )
        
        # Split data
        if validation_strategy == 'patient_split':
            # Split by patients to avoid data leakage
            unique_patients = list(set(window_patient_ids))
            train_patients, test_patients = train_test_split(
                unique_patients, test_size=test_size, random_state=42,
                stratify=[patient_labels[pid] for pid in unique_patients]
            )
            
            train_mask = np.array([pid in train_patients for pid in window_patient_ids])
            test_mask = ~train_mask
            
            X_train, X_test = X_windows[train_mask], X_windows[test_mask]
            y_train, y_test = y_windows[train_mask], y_windows[test_mask]
            ids_train = [pid for i, pid in enumerate(window_patient_ids) if train_mask[i]]
            ids_test = [pid for i, pid in enumerate(window_patient_ids) if test_mask[i]]
            
        else:  # random_split
            X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
                X_windows, y_windows, window_patient_ids,
                test_size=test_size, random_state=42, stratify=y_windows
            )
        
        print(f"Train windows: {len(X_train)}, Test windows: {len(X_test)}")
        print(f"Train patients: {len(set(ids_train))}, Test patients: {len(set(ids_test))}")
        
        # Scale features (for non-deep learning methods)
        if self.classifier_type != 'deep':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # Initialize and train classifier
        self._initialize_classifier()
        
        if self.classifier_type == 'deep':
            results = self._train_deep_classifier(
                X_train_scaled, y_train, X_test_scaled, y_test,
                ids_train, ids_test, **train_kwargs
            )
        else:
            results = self._train_sklearn_classifier(
                X_train_scaled, y_train, X_test_scaled, y_test,
                ids_train, ids_test
            )
        
        return results
    
    def _train_sklearn_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 ids_train: List[str], ids_test: List[str]) -> Dict:
        """Train sklearn-based classifier"""
        print("Training sklearn classifier...")
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Predictions
        train_pred_proba = self.classifier.predict_proba(X_train)[:, 1]
        test_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Window-level metrics
        train_auc = roc_auc_score(y_train, train_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        
        print(f"Window-level Train AUC: {train_auc:.4f}")
        print(f"Window-level Test AUC: {test_auc:.4f}")
        
        return {
            'window_train_auc': train_auc,
            'window_test_auc': test_auc,
            'train_predictions': train_pred_proba,
            'test_predictions': test_pred_proba,
            'y_train': y_train,
            'y_test': y_test,
            'ids_train': ids_train,
            'ids_test': ids_test,
            'classifier_type': self.classifier_type
        }
    
    def _train_deep_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              ids_train: List[str], ids_test: List[str],
                              epochs: int = 100, batch_size: int = 256,
                              lr: float = 0.001, **kwargs) -> Dict:
        """Train deep learning classifier"""
        print("Training deep learning classifier...")
        
        # Create datasets
        train_dataset = WindowDataset(X_train, y_train, ids_train)
        test_dataset = WindowDataset(X_test, y_test, ids_test)
        
        # Handle class imbalance with weighted sampling
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = [class_weights[int(label)] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        train_aucs = []
        test_aucs = []
        best_test_auc = 0
        
        for epoch in range(epochs):
            # Training
            self.classifier.train()
            epoch_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())
            
            # Validation
            self.classifier.eval()
            test_preds = []
            test_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.classifier(features)
                    probs = torch.sigmoid(outputs)
                    
                    test_preds.extend(probs.detach().cpu().numpy())
                    test_labels.extend(labels.detach().cpu().numpy())
            
            # Calculate metrics
            train_auc = roc_auc_score(train_labels, train_preds)
            test_auc = roc_auc_score(test_labels, test_preds)
            
            train_losses.append(epoch_loss / len(train_loader))
            train_aucs.append(train_auc)
            test_aucs.append(test_auc)
            
            scheduler.step(test_auc)
            
            # Save best model
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                torch.save(self.classifier.state_dict(), 'best_window_model.pth')
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss: {train_losses[-1]:.4f}, "
                      f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
        
        # Load best model
        self.classifier.load_state_dict(torch.load('best_window_model.pth'))
        
        # Final predictions
        self.classifier.eval()
        final_train_preds = []
        final_test_preds = []
        
        with torch.no_grad():
            # Train predictions
            train_loader_eval = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            for batch in train_loader_eval:
                features = batch['features'].to(self.device)
                outputs = self.classifier(features)
                probs = torch.sigmoid(outputs)
                final_train_preds.extend(probs.detach().cpu().numpy())
            
            # Test predictions
            for batch in test_loader:
                features = batch['features'].to(self.device)
                outputs = self.classifier(features)
                probs = torch.sigmoid(outputs)
                final_test_preds.extend(probs.detach().cpu().numpy())
        
        final_train_auc = roc_auc_score(y_train, final_train_preds)
        final_test_auc = roc_auc_score(y_test, final_test_preds)
        
        print(f"Final Window-level Train AUC: {final_train_auc:.4f}")
        print(f"Final Window-level Test AUC: {final_test_auc:.4f}")
        
        return {
            'window_train_auc': final_train_auc,
            'window_test_auc': final_test_auc,
            'train_predictions': np.array(final_train_preds).flatten(),
            'test_predictions': np.array(final_test_preds).flatten(),
            'y_train': y_train,
            'y_test': y_test,
            'ids_train': ids_train,
            'ids_test': ids_test,
            'classifier_type': self.classifier_type,
            'training_history': {
                'train_losses': train_losses,
                'train_aucs': train_aucs,
                'test_aucs': test_aucs
            }
        }
    
    def aggregate_predictions(self, window_predictions: np.ndarray,
                            window_patient_ids: List[str],
                            methods: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Aggregate window predictions to patient level
        
        Args:
            window_predictions: Array of window-level predictions
            window_patient_ids: List of patient IDs for each window
            methods: List of aggregation methods to use
            
        Returns:
            Dictionary {method: {patient_id: prediction}}
        """
        if methods is None:
            methods = [
                'mean_probability',
                'majority_vote',
                'max_probability',
                'weighted_vote',
                'threshold_based',
                'percentile_based'
            ]
        
        # Group predictions by patient
        patient_predictions = {}
        for pred, patient_id in zip(window_predictions, window_patient_ids):
            if patient_id not in patient_predictions:
                patient_predictions[patient_id] = []
            patient_predictions[patient_id].append(pred)
        
        # Convert to arrays
        for patient_id in patient_predictions:
            patient_predictions[patient_id] = np.array(patient_predictions[patient_id])
        
        results = {}
        
        for method in methods:
            method_results = {}
            
            for patient_id, preds in patient_predictions.items():
                if method == 'mean_probability':
                    agg_pred = np.mean(preds)
                
                elif method == 'majority_vote':
                    votes = (preds > 0.5).astype(int)
                    agg_pred = np.mean(votes)
                
                elif method == 'max_probability':
                    agg_pred = np.max(preds)
                
                elif method == 'weighted_vote':
                    # Weight by confidence (distance from 0.5)
                    weights = np.abs(preds - 0.5)
                    if np.sum(weights) > 0:
                        agg_pred = np.average(preds, weights=weights)
                    else:
                        agg_pred = np.mean(preds)
                
                elif method == 'threshold_based':
                    # Patient is positive if >30% of windows are positive
                    positive_windows = np.sum(preds > 0.5)
                    agg_pred = 1.0 if positive_windows / len(preds) > 0.3 else 0.0
                
                elif method == 'percentile_based':
                    # Use 75th percentile
                    agg_pred = np.percentile(preds, 75)
                
                else:
                    agg_pred = np.mean(preds)  # Default to mean
                
                method_results[patient_id] = agg_pred
            
            results[method] = method_results
        
        return results
    
    def evaluate_patient_level(self, results: Dict, patient_labels: Dict[str, int]) -> pd.DataFrame:
        """
        Evaluate patient-level performance after aggregation
        
        Args:
            results: Output from train() method
            patient_labels: True patient labels
            
        Returns:
            DataFrame with evaluation results for different aggregation methods
        """
        # Get window predictions and patient IDs
        test_preds = results['test_predictions']
        test_ids = results['ids_test']
        
        # Aggregate predictions
        aggregated_preds = self.aggregate_predictions(test_preds, test_ids)
        
        evaluation_results = []
        
        for method, patient_preds in aggregated_preds.items():
            # Get true labels and predictions for common patients
            y_true = []
            y_pred = []
            
            for patient_id, pred in patient_preds.items():
                if patient_id in patient_labels:
                    y_true.append(patient_labels[patient_id])
                    y_pred.append(pred)
            
            if len(y_true) > 0 and len(set(y_true)) > 1:  # Need both classes
                auc = roc_auc_score(y_true, y_pred)
                
                # Binary predictions for accuracy
                y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
                accuracy = np.mean(np.array(y_true) == y_pred_binary)
                
                evaluation_results.append({
                    'aggregation_method': method,
                    'patient_auc': auc,
                    'patient_accuracy': accuracy,
                    'n_patients': len(y_true)
                })
        
        df_results = pd.DataFrame(evaluation_results)
        df_results = df_results.sort_values('patient_auc', ascending=False)
        
        print(f"\n=== PATIENT-LEVEL EVALUATION ===")
        print(df_results.to_string(index=False))
        
        return df_results
    
    def predict_windows(self, window_features: np.ndarray) -> np.ndarray:
        """Predict on new window features"""
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")
        
        if self.classifier_type == 'deep':
            self.classifier.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(window_features).to(self.device)
                outputs = self.classifier(features_tensor)
                predictions = torch.sigmoid(outputs).cpu().numpy().flatten()
        else:
            features_scaled = self.scaler.transform(window_features)
            predictions = self.classifier.predict_proba(features_scaled)[:, 1]
        
        return predictions
    
    def predict_patients(self, patient_features: Dict[str, np.ndarray],
                        aggregation_method: str = 'mean_probability') -> Dict[str, float]:
        """Predict patient-level outcomes"""
        all_predictions = []
        all_patient_ids = []
        
        for patient_id, features in patient_features.items():
            if len(features) > 0:
                window_preds = self.predict_windows(features)
                all_predictions.extend(window_preds)
                all_patient_ids.extend([patient_id] * len(window_preds))
        
        # Aggregate predictions
        aggregated = self.aggregate_predictions(
            np.array(all_predictions), all_patient_ids, [aggregation_method]
        )
        
        return aggregated[aggregation_method]
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'classifier_type': self.classifier_type,
            'scaler': self.scaler,
            'patient_window_mapping': self.patient_window_mapping,
            'kwargs': self.kwargs
        }
        
        if self.classifier_type == 'deep':
            model_data['model_state_dict'] = self.classifier.state_dict()
            model_data['model_config'] = {
                'input_dim': self.classifier.classifier[0].in_features,
                'hidden_dims': [layer.out_features for layer in self.classifier.classifier 
                               if isinstance(layer, nn.Linear)][:-1]
            }
        else:
            model_data['classifier'] = self.classifier
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier_type = model_data['classifier_type']
        self.scaler = model_data['scaler']
        self.patient_window_mapping = model_data['patient_window_mapping']
        self.kwargs = model_data['kwargs']
        
        if self.classifier_type == 'deep':
            config = model_data['model_config']
            self.classifier = DeepWindowClassifier(
                input_dim=config['input_dim'],
                hidden_dims=config['hidden_dims']
            ).to(self.device)
            self.classifier.load_state_dict(model_data['model_state_dict'])
        else:
            self.classifier = model_data['classifier']
        
        print(f"Model loaded from {filepath}")
    
    def plot_results(self, results: Dict, patient_labels: Dict[str, int] = None,
                    save_path: str = None):
        """Plot training results and patient-level evaluation"""
        
        if self.classifier_type == 'deep' and 'training_history' in results:
            # Plot training curves for deep learning
            history = results['training_history']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Training loss
            axes[0, 0].plot(history['train_losses'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Training AUC
            axes[0, 1].plot(history['train_aucs'], label='Train')
            axes[0, 1].plot(history['test_aucs'], label='Test')
            axes[0, 1].set_title('Window-Level AUC')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Window-level ROC curve
        fpr, tpr, _ = roc_curve(results['y_test'], results['test_predictions'])
        axes[0, 2].plot(fpr, tpr, label=f"Window AUC: {results['window_test_auc']:.3f}")
        axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('Window-Level ROC Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Window-level confusion matrix
        y_pred_binary = (results['test_predictions'] > 0.5).astype(int)
        cm = confusion_matrix(results['y_test'], y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Window-Level Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Patient-level evaluation (if labels provided)
        if patient_labels is not None:
            patient_results = self.evaluate_patient_level(results, patient_labels)
            
            # Patient-level AUC comparison
            axes[1, 1].bar(range(len(patient_results)), patient_results['patient_auc'])
            axes[1, 1].set_xlabel('Aggregation Method')
            axes[1, 1].set_ylabel('Patient AUC')
            axes[1, 1].set_title('Patient-Level AUC by Aggregation Method')
            axes[1, 1].set_xticks(range(len(patient_results)))
            axes[1, 1].set_xticklabels(patient_results['aggregation_method'], rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Best patient-level ROC
            best_method = patient_results.iloc[0]['aggregation_method']
            aggregated_preds = self.aggregate_predictions(
                results['test_predictions'], results['ids_test'], [best_method]
            )
            
            y_true_patient = []
            y_pred_patient = []
            for patient_id, pred in aggregated_preds[best_method].items():
                if patient_id in patient_labels:
                    y_true_patient.append(patient_labels[patient_id])
                    y_pred_patient.append(pred)
            
            if len(y_true_patient) > 0 and len(set(y_true_patient)) > 1:
                fpr_patient, tpr_patient, _ = roc_curve(y_true_patient, y_pred_patient)
                patient_auc = roc_auc_score(y_true_patient, y_pred_patient)
                
                axes[1, 2].plot(fpr_patient, tpr_patient, 
                               label=f"Patient AUC ({best_method}): {patient_auc:.3f}")
                axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[1, 2].set_xlabel('False Positive Rate')
                axes[1, 2].set_ylabel('True Positive Rate')
                axes[1, 2].set_title('Best Patient-Level ROC Curve')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor patient ROC', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
        else:
            # Placeholder plots if no patient labels
            axes[1, 1].text(0.5, 0.5, 'Patient labels\nnot provided', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 2].text(0.5, 0.5, 'Patient labels\nnot provided', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        
        plt.show()

def load_patient_features(features_dir: str) -> Dict[str, np.ndarray]:
    """
    Load patient features from directory of pickle files
    
    Args:
        features_dir: Directory containing feature pickle files
        
    Returns:
        Dictionary {patient_id: features}
    """
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

def compare_label_strategies(patient_features: Dict[str, np.ndarray],
                           patient_labels: Dict[str, int],
                           classifier_type: str = 'random_forest') -> pd.DataFrame:
    """
    Compare different window labeling strategies
    
    Args:
        patient_features: Patient features dictionary
        patient_labels: Patient labels dictionary
        classifier_type: Type of classifier to use
        
    Returns:
        DataFrame with comparison results
    """
    strategies = ['inherit', 'noisy', 'activity_based']
    results = []
    
    for strategy in strategies:
        print(f"\nTesting label strategy: {strategy}")
        
        try:
            classifier = WindowLevelClassifier(classifier_type=classifier_type)
            
            result = classifier.train(
                patient_features, patient_labels,
                label_strategy=strategy,
                validation_strategy='patient_split'
            )
            
            # Evaluate patient-level performance
            patient_eval = classifier.evaluate_patient_level(result, patient_labels)
            best_patient_auc = patient_eval['patient_auc'].max() if len(patient_eval) > 0 else 0
            
            results.append({
                'strategy': strategy,
                'window_train_auc': result['window_train_auc'],
                'window_test_auc': result['window_test_auc'],
                'best_patient_auc': best_patient_auc
            })
            
        except Exception as e:
            print(f"Error with strategy {strategy}: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('best_patient_auc', ascending=False)
    
    print(f"\n=== LABEL STRATEGY COMPARISON ===")
    print(df_results.to_string(index=False))
    
    return df_results

def compare_classifiers(patient_features: Dict[str, np.ndarray],
                       patient_labels: Dict[str, int],
                       label_strategy: str = 'inherit') -> pd.DataFrame:
    """
    Compare different classifier types for window-level classification
    
    Args:
        patient_features: Patient features dictionary
        patient_labels: Patient labels dictionary
        label_strategy: Window labeling strategy
        
    Returns:
        DataFrame with comparison results
    """
    classifiers = [
        ('random_forest', {}),
        ('logistic_regression', {}),
        ('deep', {'epochs': 50, 'batch_size': 256})
    ]
    
    results = []
    
    for classifier_type, kwargs in classifiers:
        print(f"\nTesting classifier: {classifier_type}")
        
        try:
            classifier = WindowLevelClassifier(classifier_type=classifier_type)
            
            result = classifier.train(
                patient_features, patient_labels,
                label_strategy=label_strategy,
                validation_strategy='patient_split',
                **kwargs
            )
            
            # Evaluate patient-level performance
            patient_eval = classifier.evaluate_patient_level(result, patient_labels)
            best_patient_auc = patient_eval['patient_auc'].max() if len(patient_eval) > 0 else 0
            best_aggregation = patient_eval.iloc[0]['aggregation_method'] if len(patient_eval) > 0 else 'N/A'
            
            results.append({
                'classifier': classifier_type,
                'window_train_auc': result['window_train_auc'],
                'window_test_auc': result['window_test_auc'],
                'best_patient_auc': best_patient_auc,
                'best_aggregation': best_aggregation
            })
            
        except Exception as e:
            print(f"Error with classifier {classifier_type}: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('best_patient_auc', ascending=False)
    
    print(f"\n=== CLASSIFIER COMPARISON ===")
    print(df_results.to_string(index=False))
    
    return df_results

def analyze_patient_window_patterns(results: Dict, patient_labels: Dict[str, int],
                                  save_path: str = None):
    """
    Analyze patterns in window predictions across patients
    
    Args:
        results: Training results from WindowLevelClassifier
        patient_labels: True patient labels
        save_path: Path to save analysis plots
    """
    # Group predictions by patient
    patient_window_preds = {}
    for pred, patient_id in zip(results['test_predictions'], results['ids_test']):
        if patient_id not in patient_window_preds:
            patient_window_preds[patient_id] = []
        patient_window_preds[patient_id].append(pred)
    
    # Convert to arrays and get patient labels
    frail_patients = []
    nonfrail_patients = []
    
    for patient_id, preds in patient_window_preds.items():
        if patient_id in patient_labels:
            preds_array = np.array(preds)
            if patient_labels[patient_id] == 1:
                frail_patients.append(preds_array)
            else:
                nonfrail_patients.append(preds_array)
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of mean predictions per patient
    if frail_patients and nonfrail_patients:
        frail_means = [np.mean(preds) for preds in frail_patients]
        nonfrail_means = [np.mean(preds) for preds in nonfrail_patients]
        
        axes[0, 0].hist(nonfrail_means, alpha=0.7, label='Non-frail', bins=20)
        axes[0, 0].hist(frail_means, alpha=0.7, label='Frail', bins=20)
        axes[0, 0].set_xlabel('Mean Window Prediction')
        axes[0, 0].set_ylabel('Number of Patients')
        axes[0, 0].set_title('Distribution of Mean Predictions per Patient')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution of prediction variability per patient
    if frail_patients and nonfrail_patients:
        frail_stds = [np.std(preds) for preds in frail_patients]
        nonfrail_stds = [np.std(preds) for preds in nonfrail_patients]
        
        axes[0, 1].hist(nonfrail_stds, alpha=0.7, label='Non-frail', bins=20)
        axes[0, 1].hist(frail_stds, alpha=0.7, label='Frail', bins=20)
        axes[0, 1].set_xlabel('Std of Window Predictions')
        axes[0, 1].set_ylabel('Number of Patients')
        axes[0, 1].set_title('Prediction Variability per Patient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Example patient trajectories
    if frail_patients and nonfrail_patients:
        # Show first few patients from each group
        n_examples = min(3, len(frail_patients), len(nonfrail_patients))
        
        for i in range(n_examples):
            axes[0, 2].plot(frail_patients[i], alpha=0.7, color='red', 
                          label='Frail' if i == 0 else "")
            axes[0, 2].plot(nonfrail_patients[i], alpha=0.7, color='blue', 
                          label='Non-frail' if i == 0 else "")
        
        axes[0, 2].set_xlabel('Window Index')
        axes[0, 2].set_ylabel('Prediction Probability')
        axes[0, 2].set_title('Example Patient Window Trajectories')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Correlation between different aggregation methods
    test_preds = results['test_predictions']
    test_ids = results['ids_test']
    
    # Get aggregated predictions
    classifier = WindowLevelClassifier()  # Temporary instance for aggregation
    aggregated = classifier.aggregate_predictions(test_preds, test_ids)
    
    # Compare mean vs max aggregation
    mean_preds = []
    max_preds = []
    
    for patient_id in set(test_ids):
        if patient_id in aggregated['mean_probability'] and patient_id in aggregated['max_probability']:
            mean_preds.append(aggregated['mean_probability'][patient_id])
            max_preds.append(aggregated['max_probability'][patient_id])
    
    if mean_preds and max_preds:
        axes[1, 0].scatter(mean_preds, max_preds, alpha=0.6)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 0].set_xlabel('Mean Aggregation')
        axes[1, 0].set_ylabel('Max Aggregation')
        axes[1, 0].set_title('Correlation: Mean vs Max Aggregation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(mean_preds, max_preds)[0, 1]
        axes[1, 0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # 5. Number of windows per patient
    windows_per_patient = [len(preds) for preds in patient_window_preds.values()]
    
    axes[1, 1].hist(windows_per_patient, bins=20, alpha=0.7)
    axes[1, 1].set_xlabel('Number of Windows')
    axes[1, 1].set_ylabel('Number of Patients')
    axes[1, 1].set_title('Distribution of Windows per Patient')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics
    mean_windows = np.mean(windows_per_patient)
    std_windows = np.std(windows_per_patient)
    axes[1, 1].axvline(mean_windows, color='red', linestyle='--', 
                      label=f'Mean: {mean_windows:.1f}')
    axes[1, 1].legend()
    
    # 6. Aggregation method performance comparison
    if patient_labels:
        classifier_temp = WindowLevelClassifier()
        patient_eval = classifier_temp.evaluate_patient_level(results, patient_labels)
        
        if len(patient_eval) > 0:
            methods = patient_eval['aggregation_method'].values
            aucs = patient_eval['patient_auc'].values
            
            bars = axes[1, 2].bar(range(len(methods)), aucs)
            axes[1, 2].set_xlabel('Aggregation Method')
            axes[1, 2].set_ylabel('Patient AUC')
            axes[1, 2].set_title('Patient AUC by Aggregation Method')
            axes[1, 2].set_xticks(range(len(methods)))
            axes[1, 2].set_xticklabels(methods, rotation=45, ha='right')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Highlight best method
            best_idx = np.argmax(aucs)
            bars[best_idx].set_color('red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train window-level classifier")
    parser.add_argument("features_dir", help="Directory containing extracted features")
    parser.add_argument("labels_file", help="CSV file with patient labels")
    parser.add_argument("--output_dir", "-o", default="window_level_results", 
                       help="Output directory")
    parser.add_argument("--classifier", choices=['random_forest', 'logistic_regression', 'deep'],
                       default='random_forest', help="Classifier type")
    parser.add_argument("--label_strategy", choices=['inherit', 'noisy', 'activity_based'],
                       default='inherit', help="Window labeling strategy")
    parser.add_argument("--validation", choices=['random_split', 'patient_split'],
                       default='patient_split', help="Validation strategy")
    parser.add_argument("--compare_strategies", action="store_true",
                       help="Compare different labeling strategies")
    parser.add_argument("--compare_classifiers", action="store_true",
                       help="Compare different classifier types")
    parser.add_argument("--analyze_patterns", action="store_true",
                       help="Analyze patient-window patterns")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for deep learning")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for deep learning")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for deep learning")
    
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
    
    if args.compare_strategies:
        # Compare labeling strategies
        strategy_results = compare_label_strategies(
            patient_features, patient_labels, args.classifier
        )
        strategy_results.to_csv(
            os.path.join(args.output_dir, "label_strategy_comparison.csv"),
            index=False
        )
    
    if args.compare_classifiers:
        # Compare classifier types
        classifier_results = compare_classifiers(
            patient_features, patient_labels, args.label_strategy
        )
        classifier_results.to_csv(
            os.path.join(args.output_dir, "classifier_comparison.csv"),
            index=False
        )
    
    # Train final model
    print(f"\nTraining final model...")
    print(f"Classifier: {args.classifier}")
    print(f"Label strategy: {args.label_strategy}")
    print(f"Validation: {args.validation}")
    
    classifier = WindowLevelClassifier(classifier_type=args.classifier)
    
    train_kwargs = {}
    if args.classifier == 'deep':
        train_kwargs = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr
        }
    
    results = classifier.train(
        patient_features, patient_labels,
        label_strategy=args.label_strategy,
        validation_strategy=args.validation,
        **train_kwargs
    )
    
    # Evaluate patient-level performance
    patient_eval = classifier.evaluate_patient_level(results, patient_labels)
    
    # Save model
    model_path = os.path.join(args.output_dir, "window_level_model.pkl")
    classifier.save_model(model_path)
    
    # Save results
    results_path = os.path.join(args.output_dir, "training_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save patient evaluation
    patient_eval.to_csv(
        os.path.join(args.output_dir, "patient_evaluation.csv"),
        index=False
    )
    
    # Plot results
    plot_path = os.path.join(args.output_dir, "results_plot.png")
    classifier.plot_results(results, patient_labels, plot_path)
    
    # Analyze patterns if requested
    if args.analyze_patterns:
        analysis_path = os.path.join(args.output_dir, "pattern_analysis.png")
        analyze_patient_window_patterns(results, patient_labels, analysis_path)
    
    print(f"\nTraining complete! Results saved to {args.output_dir}")
    print(f"Best patient AUC: {patient_eval['patient_auc'].max():.4f} "
          f"({patient_eval.iloc[0]['aggregation_method']})")            