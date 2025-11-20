#!/usr/bin/env python3
"""
Master Script for SSL-Wearables Frailty Classification
Orchestrates the entire pipeline from CSV files to trained models
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    from feature_extraction import extract_features_from_csv, batch_extract_features
    from patient_level_classifier import PatientLevelClassifier, load_patient_features, compare_aggregation_methods
    from window_level_classifier import WindowLevelClassifier, compare_label_strategies, compare_classifiers
    from hybrid_attention_classifier import HybridAttentionTrainer, compare_all_approaches, visualize_attention_patterns
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all classifier files are in the same directory")
    sys.exit(1)

class FrailtyClassificationPipeline:
    """Complete pipeline for frailty classification"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.results = {}
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config.get('output_dir', 'results')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Frailty Classification Pipeline Started")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def validate_inputs(self) -> bool:
        """Validate input files and directories"""
        self.logger.info("Validating inputs...")
        
        # Check data input
        if self.config.get('data_type') == 'csv_directory':
            csv_dir = self.config.get('csv_directory')
            if not os.path.exists(csv_dir):
                self.logger.error(f"CSV directory not found: {csv_dir}")
                return False
            
            import glob
            csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
            if len(csv_files) == 0:
                self.logger.error(f"No CSV files found in {csv_dir}")
                return False
            
            self.logger.info(f"Found {len(csv_files)} CSV files")
            
        elif self.config.get('data_type') == 'extracted_features':
            features_dir = self.config.get('features_directory')
            if not os.path.exists(features_dir):
                self.logger.error(f"Features directory not found: {features_dir}")
                return False
        
        # Check labels file
        labels_file = self.config.get('labels_file')
        if not os.path.exists(labels_file):
            self.logger.error(f"Labels file not found: {labels_file}")
            return False
        
        # Validate labels file format
        try:
            labels_df = pd.read_csv(labels_file)
            required_cols = ['patient_id', 'label']
            
            for col in required_cols:
                if col not in labels_df.columns:
                    self.logger.error(f"Labels file missing column: {col}")
                    return False
            
            # Check label values
            unique_labels = labels_df['label'].unique()
            if not all(label in [0, 1] for label in unique_labels):
                self.logger.error("Labels must be 0 or 1 (binary classification)")
                return False
            
            self.logger.info(f"Labels file validated: {len(labels_df)} patients, "
                           f"distribution: {labels_df['label'].value_counts().to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Error reading labels file: {e}")
            return False
        
        return True
    
    def extract_features(self) -> str:
        """Extract features from CSV files"""
        if self.config.get('data_type') != 'csv_directory':
            self.logger.info("Skipping feature extraction - using pre-extracted features")
            return self.config.get('features_directory')
        
        self.logger.info("Starting feature extraction...")
        
        csv_dir = self.config['csv_directory']
        features_dir = os.path.join(self.config['output_dir'], 'extracted_features')
        
        extraction_config = self.config.get('feature_extraction', {})
        
        try:
            results = batch_extract_features(
                csv_dir=csv_dir,
                output_dir=features_dir,
                pattern="*.csv",
                window_size=extraction_config.get('window_size', 3000),
                overlap=extraction_config.get('overlap', 0.5),
                target_hz=extraction_config.get('target_hz', 100),
                batch_size=extraction_config.get('batch_size', 64)
            )
            
            self.logger.info(f"Feature extraction completed: {len(results)} patients processed")
            self.results['feature_extraction'] = {
                'n_patients': len(results),
                'output_dir': features_dir
            }
            
            return features_dir
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise
    
    def load_data(self, features_dir: str) -> tuple:
        """Load features and labels"""
        self.logger.info("Loading patient features and labels...")
        
        # Load features
        patient_features = load_patient_features(features_dir)
        
        # Load labels
        labels_df = pd.read_csv(self.config['labels_file'])
        patient_labels = dict(zip(labels_df['patient_id'], labels_df['label']))
        
        # Find intersection
        common_patients = set(patient_features.keys()) & set(patient_labels.keys())
        
        if len(common_patients) == 0:
            raise ValueError("No common patients between features and labels!")
        
        # Filter to common patients
        patient_features = {pid: patient_features[pid] for pid in common_patients}
        patient_labels = {pid: patient_labels[pid] for pid in common_patients}
        
        self.logger.info(f"Loaded data for {len(common_patients)} patients")
        self.logger.info(f"Label distribution: {pd.Series(patient_labels.values()).value_counts().to_dict()}")
        
        # Data quality check
        window_counts = [len(features) for features in patient_features.values()]
        self.logger.info(f"Windows per patient - Mean: {np.mean(window_counts):.1f}, "
                        f"Std: {np.std(window_counts):.1f}, "
                        f"Range: [{np.min(window_counts)}, {np.max(window_counts)}]")
        
        return patient_features, patient_labels
    
    def run_patient_level_classification(self, patient_features: Dict, patient_labels: Dict) -> Dict:
        """Run patient-level classification"""
        if not self.config.get('run_patient_level', True):
            self.logger.info("Skipping patient-level classification")
            return {}
        
        self.logger.info("Running patient-level classification...")
        
        patient_config = self.config.get('patient_level', {})
        results_dir = os.path.join(self.config['output_dir'], 'patient_level_results')
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # Compare aggregation methods if requested
            if patient_config.get('compare_aggregations', False):
                self.logger.info("Comparing aggregation methods...")
                comparison_results = compare_aggregation_methods(
                    patient_features, patient_labels,
                    classifier_type=patient_config.get('classifier_type', 'random_forest')
                )
                comparison_results.to_csv(
                    os.path.join(results_dir, "aggregation_comparison.csv"), index=False
                )
                best_aggregation = comparison_results.iloc[0]['method']
            else:
                best_aggregation = patient_config.get('aggregation_method', 'comprehensive')
            
            # Train final model
            classifier = PatientLevelClassifier(
                classifier_type=patient_config.get('classifier_type', 'random_forest')
            )
            
            results = classifier.train(
                patient_features, patient_labels,
                aggregation_method=best_aggregation,
                test_size=patient_config.get('test_size', 0.2),
                optimize_hyperparameters=patient_config.get('optimize_hyperparameters', True)
            )
            
            # Save model and results
            classifier.save_model(os.path.join(results_dir, "patient_level_model.pkl"))
            
            # Plot results
            classifier.plot_results(results, os.path.join(results_dir, "results_plot.png"))
            
            self.logger.info(f"Patient-level classification completed - AUC: {results['test_auc']:.4f}")
            
            return {
                'test_auc': results['test_auc'],
                'cv_auc': results['cv_auc_mean'],
                'aggregation_method': best_aggregation,
                'results_dir': results_dir
            }
            
        except Exception as e:
            self.logger.error(f"Patient-level classification failed: {e}")
            return {'error': str(e)}
    
    def run_window_level_classification(self, patient_features: Dict, patient_labels: Dict) -> Dict:
        """Run window-level classification"""
        if not self.config.get('run_window_level', True):
            self.logger.info("Skipping window-level classification")
            return {}
        
        self.logger.info("Running window-level classification...")
        
        window_config = self.config.get('window_level', {})
        results_dir = os.path.join(self.config['output_dir'], 'window_level_results')
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # Compare strategies if requested
            if window_config.get('compare_strategies', False):
                self.logger.info("Comparing labeling strategies...")
                strategy_results = compare_label_strategies(
                    patient_features, patient_labels,
                    classifier_type=window_config.get('classifier_type', 'random_forest')
                )
                strategy_results.to_csv(
                    os.path.join(results_dir, "strategy_comparison.csv"), index=False
                )
                best_strategy = strategy_results.iloc[0]['strategy']
            else:
                best_strategy = window_config.get('label_strategy', 'inherit')
            
            # Compare classifiers if requested
            if window_config.get('compare_classifiers', False):
                self.logger.info("Comparing classifier types...")
                classifier_results = compare_classifiers(
                    patient_features, patient_labels, best_strategy
                )
                classifier_results.to_csv(
                    os.path.join(results_dir, "classifier_comparison.csv"), index=False
                )
                best_classifier = classifier_results.iloc[0]['classifier']
            else:
                best_classifier = window_config.get('classifier_type', 'random_forest')
            
            # Train final model
            classifier = WindowLevelClassifier(classifier_type=best_classifier)
            
            train_kwargs = {}
            if best_classifier == 'deep':
                train_kwargs = {
                    'epochs': window_config.get('epochs', 100),
                    'batch_size': window_config.get('batch_size', 256),
                    'lr': window_config.get('lr', 0.001)
                }
            
            results = classifier.train(
                patient_features, patient_labels,
                label_strategy=best_strategy,
                validation_strategy=window_config.get('validation_strategy', 'patient_split'),
                **train_kwargs
            )
            
            # Evaluate patient-level performance
            patient_eval = classifier.evaluate_patient_level(results, patient_labels)
            best_patient_auc = patient_eval['patient_auc'].max()
            best_aggregation = patient_eval.iloc[0]['aggregation_method']
            
            # Save model and results
            classifier.save_model(os.path.join(results_dir, "window_level_model.pkl"))
            
            # Plot results
            classifier.plot_results(results, patient_labels, 
                                  os.path.join(results_dir, "results_plot.png"))
            
            # Save patient evaluation
            patient_eval.to_csv(os.path.join(results_dir, "patient_evaluation.csv"), index=False)
            
            self.logger.info(f"Window-level classification completed - "
                           f"Window AUC: {results['window_test_auc']:.4f}, "
                           f"Best Patient AUC: {best_patient_auc:.4f}")
            
            return {
                'window_auc': results['window_test_auc'],
                'best_patient_auc': best_patient_auc,
                'best_aggregation': best_aggregation,
                'label_strategy': best_strategy,
                'classifier_type': best_classifier,
                'results_dir': results_dir
            }
            
        except Exception as e:
            self.logger.error(f"Window-level classification failed: {e}")
            return {'error': str(e)}
    
    def run_hybrid_attention_classification(self, patient_features: Dict, patient_labels: Dict) -> Dict:
        """Run hybrid attention classification"""
        if not self.config.get('run_hybrid', True):
            self.logger.info("Skipping hybrid attention classification")
            return {}
        
        self.logger.info("Running hybrid attention classification...")
        
        hybrid_config = self.config.get('hybrid_attention', {})
        results_dir = os.path.join(self.config['output_dir'], 'hybrid_results')
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # Load or create SSL backbone
            ssl_model_path = hybrid_config.get('ssl_model_path')
            if ssl_model_path and os.path.exists(ssl_model_path):
                import torch
                ssl_backbone = torch.load(ssl_model_path)
                self.logger.info(f"Loaded SSL backbone from {ssl_model_path}")
            else:
                from feature_extraction import SSLWearablesBackbone
                ssl_backbone = SSLWearablesBackbone(n_features=1024)
                self.logger.info("Using placeholder SSL backbone")
            
            # Initialize trainer
            trainer = HybridAttentionTrainer(
                ssl_backbone,
                freeze_backbone=hybrid_config.get('freeze_backbone', False),
                projection_dim=hybrid_config.get('projection_dim', 512),
                n_attention_layers=hybrid_config.get('n_attention_layers', 2),
                n_heads=hybrid_config.get('n_heads', 8),
                dropout=hybrid_config.get('dropout', 0.1)
            )
            
            # Train model
            results = trainer.train(
                patient_features, patient_labels,
                epochs=hybrid_config.get('epochs', 100),
                batch_size=hybrid_config.get('batch_size', 8),
                lr=hybrid_config.get('lr', 0.001),
                early_stopping_patience=hybrid_config.get('early_stopping_patience', 15)
            )
            
            # Save model
            trainer.save_model(os.path.join(results_dir, "hybrid_attention_model.pth"))
            
            # Visualize attention if requested
            if (hybrid_config.get('visualize_attention', True) and 
                results.get('test_attention_weights')):
                
                attention_dict = {
                    pid: weights for pid, weights in 
                    zip(results['test_patient_ids'], results['test_attention_weights'])
                }
                
                visualize_attention_patterns(
                    attention_dict, patient_labels,
                    os.path.join(results_dir, "attention_patterns.png")
                )
            
            self.logger.info(f"Hybrid attention classification completed - "
                           f"AUC: {results['test_auc']:.4f}, "
                           f"Best AUC: {results['best_test_auc']:.4f}")
            
            return {
                'test_auc': results['test_auc'],
                'best_test_auc': results['best_test_auc'],
                'val_auc': results.get('val_auc'),
                'results_dir': results_dir
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid attention classification failed: {e}")
            return {'error': str(e)}
    
    def compare_all_approaches(self, patient_features: Dict, patient_labels: Dict) -> Dict:
        """Compare all approaches"""
        if not self.config.get('compare_approaches', False):
            return {}
        
        self.logger.info("Running comprehensive approach comparison...")
        
        try:
            # Load SSL backbone
            hybrid_config = self.config.get('hybrid_attention', {})
            ssl_model_path = hybrid_config.get('ssl_model_path')
            
            if ssl_model_path and os.path.exists(ssl_model_path):
                import torch
                ssl_backbone = torch.load(ssl_model_path)
            else:
                from feature_extraction import SSLWearablesBackbone
                ssl_backbone = SSLWearablesBackbone(n_features=1024)
            
            comparison_dir = os.path.join(self.config['output_dir'], 'approach_comparison')
            
            comparison_results = compare_all_approaches(
                patient_features, patient_labels, ssl_backbone, comparison_dir
            )
            
            if len(comparison_results) > 0:
                winner = comparison_results.iloc[0]
                self.logger.info(f"Comparison completed - Winner: {winner['approach']} "
                               f"({winner['method']}) with AUC: {winner['auc']:.4f}")
                
                return {
                    'winner': winner['approach'],
                    'winner_method': winner['method'],
                    'winner_auc': winner['auc'],
                    'comparison_results': comparison_results.to_dict('records'),
                    'results_dir': comparison_dir
                }
            else:
                return {'error': 'No successful comparisons'}
                
        except Exception as e:
            self.logger.error(f"Approach comparison failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        self.logger.info("Generating final report...")
        
        report_path = os.path.join(self.config['output_dir'], 'final_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Frailty Classification Pipeline Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2))
            f.write("\n```\n\n")
            
            # Data Summary
            if 'feature_extraction' in self.results:
                f.write("## Data Summary\n\n")
                fe_results = self.results['feature_extraction']
                f.write(f"- **Patients processed:** {fe_results['n_patients']}\n")
                f.write(f"- **Features directory:** {fe_results['output_dir']}\n\n")
            
            # Results Summary
            f.write("## Results Summary\n\n")
            
            # Patient-level results
            if 'patient_level' in self.results and 'error' not in self.results['patient_level']:
                pl_results = self.results['patient_level']
                f.write("### Patient-Level Classification\n")
                f.write(f"- **Test AUC:** {pl_results['test_auc']:.4f}\n")
                f.write(f"- **CV AUC:** {pl_results['cv_auc']:.4f}\n")
                f.write(f"- **Aggregation method:** {pl_results['aggregation_method']}\n\n")
            
            # Window-level results
            if 'window_level' in self.results and 'error' not in self.results['window_level']:
                wl_results = self.results['window_level']
                f.write("### Window-Level Classification\n")
                f.write(f"- **Window AUC:** {wl_results['window_auc']:.4f}\n")
                f.write(f"- **Best Patient AUC:** {wl_results['best_patient_auc']:.4f}\n")
                f.write(f"- **Best aggregation:** {wl_results['best_aggregation']}\n")
                f.write(f"- **Label strategy:** {wl_results['label_strategy']}\n")
                f.write(f"- **Classifier type:** {wl_results['classifier_type']}\n\n")
            
            # Hybrid results
            if 'hybrid' in self.results and 'error' not in self.results['hybrid']:
                h_results = self.results['hybrid']
                f.write("### Hybrid Attention Classification\n")
                f.write(f"- **Test AUC:** {h_results['test_auc']:.4f}\n")
                f.write(f"- **Best AUC:** {h_results['best_test_auc']:.4f}\n")
                if h_results.get('val_auc'):
                    f.write(f"- **Validation AUC:** {h_results['val_auc']:.4f}\n")
                f.write("\n")
            
            # Comparison results
            if 'comparison' in self.results and 'error' not in self.results['comparison']:
                comp_results = self.results['comparison']
                f.write("### Approach Comparison\n")
                f.write(f"- **Winner:** {comp_results['winner']} ({comp_results['winner_method']})\n")
                f.write(f"- **Winner AUC:** {comp_results['winner_auc']:.4f}\n\n")
                
                f.write("#### All Results\n")
                for result in comp_results['comparison_results']:
                    f.write(f"- **{result['approach']}:** {result['auc']:.4f} ({result['method']})\n")
                f.write("\n")
            
            # Errors
            errors = []
            for key, value in self.results.items():
                if isinstance(value, dict) and 'error' in value:
                    errors.append(f"- **{key}:** {value['error']}")
            
            if errors:
                f.write("## Errors\n\n")
                f.write("\n".join(errors))
                f.write("\n\n")
            
            # File locations
            f.write("## Output Files\n\n")
            f.write(f"- **Main output directory:** {self.config['output_dir']}\n")
            
            for key, value in self.results.items():
                if isinstance(value, dict) and 'results_dir' in value:
                    f.write(f"- **{key} results:** {value['results_dir']}\n")
        
        self.logger.info(f"Report generated: {report_path}")
        return report_path
    
    def run(self) -> Dict:
        """Run the complete pipeline"""
        try:
            # Validate inputs
            if not self.validate_inputs():
                raise ValueError("Input validation failed")
            
            # Extract features
            features_dir = self.extract_features()
            
            # Load data
            patient_features, patient_labels = self.load_data(features_dir)
            
            # Run classifications
            self.results['patient_level'] = self.run_patient_level_classification(
                patient_features, patient_labels
            )
            
            self.results['window_level'] = self.run_window_level_classification(
                patient_features, patient_labels
            )
            
            self.results['hybrid'] = self.run_hybrid_attention_classification(
                patient_features, patient_labels
            )
            
            # Compare approaches
            self.results['comparison'] = self.compare_all_approaches(
                patient_features, patient_labels
            )
            
            # Generate report
            report_path = self.generate_report()
            
            self.logger.info("Pipeline completed successfully!")
            
            return {
                'status': 'success',
                'results': self.results,
                'report_path': report_path
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'results': self.results
            }

def create_default_config() -> Dict:
    """Create default configuration"""
    return {
        # Data configuration
        'data_type': 'csv_directory',  # 'csv_directory' or 'extracted_features'
        'csv_directory': None,  # Required if data_type is 'csv_directory'
        'features_directory': None,  # Required if data_type is 'extracted_features'
        'labels_file': None,  # Required: CSV with patient_id, label columns
        'output_dir': 'pipeline_results',
        
        # Feature extraction configuration
        'feature_extraction': {
            'window_size': 3000,  # 30 seconds at 100Hz
            'overlap': 0.5,
            'target_hz': 100,
            'batch_size': 64
        },
        
        # Which approaches to run
        'run_patient_level': True,
        'run_window_level': True,
        'run_hybrid': True,
        'compare_approaches': True,
        
        # Patient-level configuration
        'patient_level': {
            'classifier_type': 'random_forest',  # 'random_forest' or 'logistic_regression'
            'aggregation_method': 'comprehensive',
            'compare_aggregations': True,
            'optimize_hyperparameters': True,
            'test_size': 0.2
        },
        
        # Window-level configuration
        'window_level': {
            'classifier_type': 'random_forest',  # 'random_forest', 'logistic_regression', 'deep'
            'label_strategy': 'inherit',  # 'inherit', 'noisy', 'activity_based'
            'validation_strategy': 'patient_split',  # 'patient_split' or 'random_split'
            'compare_strategies': True,
            'compare_classifiers': True,
            'epochs': 100,  # For deep learning
            'batch_size': 256,
            'lr': 0.001
        },
        
        # Hybrid attention configuration
        'hybrid_attention': {
            'ssl_model_path': None,  # Path to SSL backbone, None for placeholder
            'freeze_backbone': False,
            'projection_dim': 512,
            'n_attention_layers': 2,
            'n_heads': 8,
            'dropout': 0.1,
            'epochs': 100,
            'batch_size': 8,
            'lr': 0.001,
            'early_stopping_patience': 15,
            'visualize_attention': True
        }
    }

def main():
    parser = argparse.ArgumentParser(
        description="Complete SSL-Wearables Frailty Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline from CSV files
  python run_frailty_classification.py --csv_dir /path/to/csvs --labels /path/to/labels.csv

  # Run from pre-extracted features
  python run_frailty_classification.py --features_dir /path/to/features --labels /path/to/labels.csv

  # Use custom configuration
  python run_frailty_classification.py --config config.json

  # Quick test with only patient-level
  python run_frailty_classification.py --csv_dir /path/to/csvs --labels /path/to/labels.csv --quick
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv_dir', help='Directory containing CSV files')
    input_group.add_argument('--features_dir', help='Directory containing extracted features')
    input_group.add_argument('--config', help='JSON configuration file')
    
    # Required arguments
    parser.add_argument('--labels', help='CSV file with patient labels (patient_id, label)')
    parser.add_argument('--output_dir', '-o', default='pipeline_results', 
                       help='Output directory')
    
    # Pipeline options
    parser.add_argument('--quick', action='store_true',
                       help='Quick run (patient-level only, no comparisons)')
    parser.add_argument('--no_patient_level', action='store_true',
                       help='Skip patient-level classification')
    parser.add_argument('--no_window_level', action='store_true',
                       help='Skip window-level classification')
    parser.add_argument('--no_hybrid', action='store_true',
                       help='Skip hybrid attention classification')
    parser.add_argument('--no_comparison', action='store_true',
                       help='Skip approach comparison')
    
    # Feature extraction options
    parser.add_argument('--window_size', type=int, default=3000,
                       help='Window size in samples')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap fraction')
    parser.add_argument('--target_hz', type=int, default=100,
                       help='Target sampling rate')
    
    # Model options
    parser.add_argument('--ssl_model', help='Path to SSL backbone model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs for deep models')
    
    # Utility options
    parser.add_argument('--create_config', help='Create default config file and exit')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate inputs and exit')
    
    args = parser.parse_args()
    
    # Create default config file
    if args.create_config:
        config = create_default_config()
        with open(args.create_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default configuration saved to {args.create_config}")
        return
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
        
        # Update config from command line arguments
        if args.csv_dir:
            config['data_type'] = 'csv_directory'
            config['csv_directory'] = args.csv_dir
        elif args.features_dir:
            config['data_type'] = 'extracted_features'
            config['features_directory'] = args.features_dir
        
        if args.labels:
            config['labels_file'] = args.labels
        
        config['output_dir'] = args.output_dir
        
        # Pipeline options
        if args.quick:
            config['run_window_level'] = False
            config['run_hybrid'] = False
            config['compare_approaches'] = False
            config['patient_level']['compare_aggregations'] = False
            config['patient_level']['optimize_hyperparameters'] = False
        
        if args.no_patient_level:
            config['run_patient_level'] = False
        if args.no_window_level:
            config['run_window_level'] = False
        if args.no_hybrid:
            config['run_hybrid'] = False
        if args.no_comparison:
            config['compare_approaches'] = False
        
        # Feature extraction options
        config['feature_extraction']['window_size'] = args.window_size
        config['feature_extraction']['overlap'] = args.overlap
        config['feature_extraction']['target_hz'] = args.target_hz
        
        # Model options
        if args.ssl_model:
            config['hybrid_attention']['ssl_model_path'] = args.ssl_model
        
        config['window_level']['epochs'] = args.epochs
        config['hybrid_attention']['epochs'] = args.epochs
    
    # Validate required arguments
    if not config.get('labels_file'):
        parser.error("Labels file is required (--labels or in config)")
    
    if (config.get('data_type') == 'csv_directory' and 
        not config.get('csv_directory')):
        parser.error("CSV directory is required when using CSV input")
    
    if (config.get('data_type') == 'extracted_features' and 
        not config.get('features_directory')):
        parser.error("Features directory is required when using extracted features")
    
    # Initialize and run pipeline
    pipeline = FrailtyClassificationPipeline(config)
    
    # Validation only mode
    if args.validate_only:
        if pipeline.validate_inputs():
            print("✅ All inputs validated successfully!")
            return 0
        else:
            print("❌ Input validation failed!")
            return 1
    
    # Run pipeline
    print("🚀 Starting Frailty Classification Pipeline...")
    print(f"📁 Output directory: {config['output_dir']}")
    
    result = pipeline.run()
    
    if result['status'] == 'success':
        print("\n🎉 Pipeline completed successfully!")
        print(f"📊 Report: {result['report_path']}")
        
        # Print summary
        results = result['results']
        print("\n📈 Summary:")
        
        if 'patient_level' in results and 'error' not in results['patient_level']:
            print(f"   Patient-Level AUC: {results['patient_level']['test_auc']:.4f}")
        
        if 'window_level' in results and 'error' not in results['window_level']:
            print(f"   Window-Level AUC: {results['window_level']['best_patient_auc']:.4f}")
        
        if 'hybrid' in results and 'error' not in results['hybrid']:
            print(f"   Hybrid Attention AUC: {results['hybrid']['test_auc']:.4f}")
        
        if 'comparison' in results and 'error' not in results['comparison']:
            comp = results['comparison']
            print(f"   🏆 Winner: {comp['winner']} (AUC: {comp['winner_auc']:.4f})")
        
        return 0
    else:
        print(f"\n❌ Pipeline failed: {result['error']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())        