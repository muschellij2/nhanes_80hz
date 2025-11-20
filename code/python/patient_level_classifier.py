"""
Patient-Level Classifier: Aggregate features then classify
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FeatureAggregator:
    """Aggregate window-level features to patient level"""
    
    def __init__(self):
        self.aggregation_methods = {
            'basic': self._basic_aggregation,
            'statistical': self._statistical_aggregation,
            'percentiles': self._percentile_aggregation,
            'comprehensive': self._comprehensive_aggregation,
            'temporal': self._temporal_aggregation
        }
    
    def aggregate(self, features: np.ndarray, method: str = 'comprehensive') -> np.ndarray:
        """
        Aggregate window features to patient level
        
        Args:
            features: (n_windows, n_features) array
            method: Aggregation method
            
        Returns:
            Aggregated features
        """
        if method not in self.aggregation_methods:
            raise ValueError(f"Unknown method: {method}")
        
        return self.aggregation_methods[method](features)
    
    def _basic_aggregation(self, features: np.ndarray) -> np.ndarray:
        """Basic mean aggregation"""
        return np.mean(features, axis=0)
    
    def _statistical_aggregation(self, features: np.ndarray) -> np.ndarray:
        """Statistical moments aggregation"""
        stats = []
        stats.append(np.mean(features, axis=0))      # Mean
        stats.append(np.std(features, axis=0))       # Standard deviation
        stats.append(np.median(features, axis=0))    # Median
        stats.append(self._safe_skew(features))      # Skewness
        stats.append(self._safe_kurtosis(features))  # Kurtosis
        return np.concatenate(stats)
    
    def _percentile_aggregation(self, features: np.ndarray) -> np.ndarray:
        """Percentile-based aggregation"""
        percentiles = [10, 25, 50, 75, 90]
        stats = [np.mean(features, axis=0), np.std(features, axis=0)]
        
        for p in percentiles:
            stats.append(np.percentile(features, p, axis=0))
        
        return np.concatenate(stats)
    
    def _comprehensive_aggregation(self, features: np.ndarray) -> np.ndarray:
        """Comprehensive statistical aggregation"""
        stats = []
        
        # Central tendency
        stats.append(np.mean(features, axis=0))
        stats.append(np.median(features, axis=0))
        
        # Variability
        stats.append(np.std(features, axis=0))
        stats.append(np.percentile(features, 25, axis=0))
        stats.append(np.percentile(features, 75, axis=0))
        stats.append(np.percentile(features, 10, axis=0))
        stats.append(np.percentile(features, 90, axis=0))
        
        # Shape
        stats.append(self._safe_skew(features))
        stats.append(self._safe_kurtosis(features))
        
        # Range
        stats.append(np.min(features, axis=0))
        stats.append(np.max(features, axis=0))
        
        return np.concatenate(stats)
    
    def _temporal_aggregation(self, features: np.ndarray) -> np.ndarray:
        """Temporal pattern aggregation"""
        stats = []
        
        # Basic statistics
        stats.append(np.mean(features, axis=0))
        stats.append(np.std(features, axis=0))
        
        # Temporal trends
        trends = []
        for i in range(features.shape[1]):
            x = np.arange(len(features))
            try:
                trend = np.polyfit(x, features[:, i], 1)[0]
                trends.append(trend)
            except:
                trends.append(0)
        stats.append(np.array(trends))
        
        # Autocorrelation
        autocorrs = []
        for i in range(features.shape[1]):
            if len(features) > 1:
                try:
                    autocorr = np.corrcoef(features[:-1, i], features[1:, i])[0, 1]
                    autocorrs.append(autocorr if not np.isnan(autocorr) else 0)
                except:
                    autocorrs.append(0)
            else:
                autocorrs.append(0)
        stats.append(np.array(autocorrs))
        
        return np.concatenate(stats)
    
    def _safe_skew(self, data: np.ndarray) -> np.ndarray:
        """Compute skewness safely"""
        result = []
        for i in range(data.shape[1]):
            try:
                sk = stats.skew(data[:, i])
                result.append(sk if not np.isnan(sk) else 0)
            except:
                result.append(0)
        return np.array(result)
    
    def _safe_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Compute kurtosis safely"""
        result = []
        for i in range(data.shape[1]):
            try:
                kurt = stats.kurtosis(data[:, i])
                result.append(kurt if not np.isnan(kurt) else 0)
            except:
                result.append(0)
        return np.array(result)

class PatientLevelClassifier:
    """Patient-level classifier with feature aggregation"""
    
    def __init__(self, classifier_type: str = 'random_forest'):
        self.classifier_type = classifier_type
        self.aggregator = FeatureAggregator()
        self.scaler = StandardScaler()
        self.classifier = None
        self.feature_names = None
        
    def _initialize_classifier(self, **kwargs):
        """Initialize classifier based on type"""
        if self.classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=kwargs.get('C', 1.0)
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def prepare_data(self, patient_features: Dict[str, np.ndarray], 
                    patient_labels: Dict[str, int],
                    aggregation_method: str = 'comprehensive') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data by aggregating features
        
        Args:
            patient_features: Dict {patient_id: features(n_windows, n_features)}
            patient_labels: Dict {patient_id: label}
            aggregation_method: Method for aggregating features
            
        Returns:
            X: Aggregated features
            y: Labels
            patient_ids: Patient identifiers
        """
        X, y, patient_ids = [], [], []
        
        print(f"Aggregating features using method: {aggregation_method}")
        
        for patient_id, features in tqdm(patient_features.items(), desc="Aggregating"):
            if patient_id in patient_labels and len(features) > 0:
                # Aggregate features
                agg_features = self.aggregator.aggregate(features, aggregation_method)
                
                X.append(agg_features)
                y.append(patient_labels[patient_id])
                patient_ids.append(patient_id)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Prepared data: {X.shape[0]} patients, {X.shape[1]} features")
        print(f"Label distribution: {np.bincount(y)}")
        
        return X, y, patient_ids
    
    def train(self, patient_features: Dict[str, np.ndarray],
             patient_labels: Dict[str, int],
             aggregation_method: str = 'comprehensive',
             test_size: float = 0.2,
             optimize_hyperparameters: bool = True,
             **classifier_kwargs) -> Dict:
        """
        Train the patient-level classifier
        
        Args:
            patient_features: Patient features dictionary
            patient_labels: Patient labels dictionary
            aggregation_method: Feature aggregation method
            test_size: Test set proportion
            optimize_hyperparameters: Whether to optimize hyperparameters
            **classifier_kwargs: Additional classifier arguments
            
        Returns:
            Training results dictionary
        """
        # Prepare data
        X, y, patient_ids = self.prepare_data(
            patient_features, patient_labels, aggregation_method
        )
        
        # Split data
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, patient_ids, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize classifier
        self._initialize_classifier(**classifier_kwargs)
        
        # Hyperparameter optimization
        if optimize_hyperparameters:
            print("Optimizing hyperparameters...")
            self.classifier = self._optimize_hyperparameters(
                X_train_scaled, y_train
            )
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred_proba = self.classifier.predict_proba(X_train_scaled)[:, 1]
        test_pred_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier, X_train_scaled, y_train, 
            cv=5, scoring='roc_auc'
        )
        
        print(f"\nResults:")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = None
        if hasattr(self.classifier, 'feature_importances_'):
            feature_importance = self.classifier.feature_importances_
        
        results = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_predictions': train_pred_proba,
            'test_predictions': test_pred_proba,
            'patient_ids_train': ids_train,
            'patient_ids_test': ids_test
        }
        
        return results
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Optimize hyperparameters using GridSearchCV"""
        if self.classifier_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.classifier_type == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        else:
            return self.classifier
        
        grid_search = GridSearchCV(
            self.classifier, param_grid, 
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict(self, patient_features: Dict[str, np.ndarray],
               aggregation_method: str = 'comprehensive') -> Dict[str, float]:
        """
        Make predictions on new patients
        
        Args:
            patient_features: Patient features dictionary
            aggregation_method: Feature aggregation method
            
        Returns:
            Dictionary of predictions {patient_id: probability}
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")
        
        predictions = {}
        
        for patient_id, features in patient_features.items():
            if len(features) > 0:
                # Aggregate features
                agg_features = self.aggregator.aggregate(features, aggregation_method)
                
                # Scale features
                agg_features_scaled = self.scaler.transform(agg_features.reshape(1, -1))
                
                # Predict
                prob = self.classifier.predict_proba(agg_features_scaled)[0, 1]
                predictions[patient_id] = prob
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'classifier_type': self.classifier_type,
            'aggregator': self.aggregator
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.classifier_type = model_data['classifier_type']
        self.aggregator = model_data['aggregator']
        
        print(f"Model loaded from {filepath}")
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot training results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC curves
        from sklearn.metrics import roc_curve
        
        # Train ROC
        fpr_train, tpr_train, _ = roc_curve(results['y_train'], results['train_predictions'])
        axes[0, 0].plot(fpr_train, tpr_train, label=f"Train AUC: {results['train_auc']:.3f}")
        
        # Test ROC
        fpr_test, tpr_test, _ = roc_curve(results['y_test'], results['test_predictions'])
        axes[0, 0].plot(fpr_test, tpr_test, label=f"Test AUC: {results['test_auc']:.3f}")
        
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confusion matrices
        from sklearn.metrics import confusion_matrix
        
        # Train confusion matrix
        cm_train = confusion_matrix(results['y_train'], results['train_predictions'] > 0.5)
        sns.heatmap(cm_train, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
        axes[0, 1].set_title('Train Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # Test confusion matrix
        cm_test = confusion_matrix(results['y_test'], results['test_predictions'] > 0.5)
        sns.heatmap(cm_test, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Test Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Feature importance (if available)
        if results['feature_importance'] is not None:
            importance = results['feature_importance']
            top_indices = np.argsort(importance)[-20:]  # Top 20 features
            
            axes[1, 1].barh(range(len(top_indices)), importance[top_indices])
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Top 20 Feature Importances')
            axes[1, 1].set_yticks(range(len(top_indices)))
            axes[1, 1].set_yticklabels([f'Feature {i}' for i in top_indices])
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
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

def compare_aggregation_methods(patient_features: Dict[str, np.ndarray],
                               patient_labels: Dict[str, int],
                               classifier_type: str = 'random_forest') -> pd.DataFrame:
    """
    Compare different aggregation methods
    
    Args:
        patient_features: Patient features dictionary
        patient_labels: Patient labels dictionary
        classifier_type: Type of classifier to use
        
    Returns:
        DataFrame with comparison results
    """
    methods = ['basic', 'statistical', 'percentiles', 'comprehensive', 'temporal']
    results = []
    
    for method in methods:
        print(f"\nTesting aggregation method: {method}")
        
        try:
            classifier = PatientLevelClassifier(classifier_type=classifier_type)
            
            result = classifier.train(
                patient_features, patient_labels,
                aggregation_method=method,
                optimize_hyperparameters=False  # Skip for comparison speed
            )
            
            results.append({
                'method': method,
                'train_auc': result['train_auc'],
                'test_auc': result['test_auc'],
                'cv_auc_mean': result['cv_auc_mean'],
                'cv_auc_std': result['cv_auc_std']
            })
            
        except Exception as e:
            print(f"Error with method {method}: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_auc', ascending=False)
    
    print(f"\n=== AGGREGATION METHOD COMPARISON ===")
    print(df_results.to_string(index=False))
    
    return df_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train patient-level classifier")
    parser.add_argument("features_dir", help="Directory containing extracted features")
    parser.add_argument("labels_file", help="CSV file with patient labels")
    parser.add_argument("--output_dir", "-o", default="patient_level_results", 
                       help="Output directory")
    parser.add_argument("--classifier", choices=['random_forest', 'logistic_regression'],
                       default='random_forest', help="Classifier type")
    parser.add_argument("--aggregation", default='comprehensive',
                       help="Aggregation method")
    parser.add_argument("--compare_methods", action="store_true",
                       help="Compare different aggregation methods")
    parser.add_argument("--optimize", action="store_true",
                       help="Optimize hyperparameters")
    
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
    
    if args.compare_methods:
        # Compare aggregation methods
        comparison_results = compare_aggregation_methods(
            patient_features, patient_labels, args.classifier
        )
        
        comparison_results.to_csv(
            os.path.join(args.output_dir, "aggregation_comparison.csv"),
            index=False
        )
    
    # Train final model
    print(f"\nTraining final model with {args.aggregation} aggregation...")
    classifier = PatientLevelClassifier(classifier_type=args.classifier)
    
    results = classifier.train(
        patient_features, patient_labels,
        aggregation_method=args.aggregation,
        optimize_hyperparameters=args.optimize
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, "patient_level_model.pkl")
    classifier.save_model(model_path)
    
    # Save results
    results_path = os.path.join(args.output_dir, "training_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Plot results
    plot_path = os.path.join(args.output_dir, "results_plot.png")
    classifier.plot_results(results, plot_path)
    
    print(f"\nTraining complete! Results saved to {args.output_dir}")