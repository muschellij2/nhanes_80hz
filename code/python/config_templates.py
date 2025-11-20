"""
Configuration templates for different use cases
"""

def get_quick_test_config():
    """Configuration for quick testing"""
    return {
        'data_type': 'csv_directory',
        'output_dir': 'quick_test_results',
        'run_patient_level': True,
        'run_window_level': False,
        'run_hybrid': False,
        'compare_approaches': False,
        'feature_extraction': {
            'window_size': 1500,  # Smaller windows for speed
            'overlap': 0.5,
            'target_hz': 50,  # Lower sampling rate
            'batch_size': 32
        },
        'patient_level': {
            'classifier_type': 'random_forest',
            'aggregation_method': 'basic',
            'compare_aggregations': False,
            'optimize_hyperparameters': False,
            'test_size': 0.3
        }
    }

def get_comprehensive_config():
    """Configuration for comprehensive analysis"""
    return {
        'data_type': 'csv_directory',
        'output_dir': 'comprehensive_results',
        'run_patient_level': True,
        'run_window_level': True,
        'run_hybrid': True,
        'compare_approaches': True,
        'feature_extraction': {
            'window_size': 3000,
            'overlap': 0.5,
            'target_hz': 100,
            'batch_size': 64
        },
        'patient_level': {
            'classifier_type': 'random_forest',
            'aggregation_method': 'comprehensive',
            'compare_aggregations': True,
            'optimize_hyperparameters': True,
            'test_size': 0.2
        },
        'window_level': {
            'classifier_type': 'random_forest',
            'label_strategy': 'inherit',
            'validation_strategy': 'patient_split',
            'compare_strategies': True,
            'compare_classifiers': True
        },
        'hybrid_attention': {
            'freeze_backbone': False,
            'projection_dim': 512,
            'n_attention_layers': 3,
            'n_heads': 8,
            'epochs': 150,
            'batch_size': 8,
            'early_stopping_patience': 20,
            'visualize_attention': True
        }
    }

def get_production_config():
    """Configuration for production deployment"""
    return {
        'data_type': 'extracted_features',
        'output_dir': 'production_results',
        'run_patient_level': False,
        'run_window_level': False,
        'run_hybrid': True,
        'compare_approaches': False,
        'hybrid_attention': {
            'ssl_model_path': 'models/ssl_backbone.pth',
            'freeze_backbone': True,
            'projection_dim': 256,
            'n_attention_layers': 2,
            'n_heads': 8,
            'epochs': 50,
            'batch_size': 16,
            'early_stopping_patience': 10,
            'visualize_attention': False
        }
    }