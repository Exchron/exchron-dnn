#!/usr/bin/env python3
"""
Complete training pipeline for the exoplanet classification project.
This script orchestrates the entire process from data loading to model evaluation.
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def main():
    """
    Main training pipeline for the exoplanet classification model.
    """
    print("="*60)
    print("EXOPLANET CLASSIFICATION PROJECT")
    print("Dual-Input DNN Training Pipeline")
    print("="*60)
    
    try:
        # Import after adding src to path
        from config import config
        from data.preprocessing import load_and_preprocess_data
        
        # Check if TensorFlow is available
        try:
            import tensorflow as tf
            print(f"TensorFlow version: {tf.__version__}")
            print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
            
            # Set memory growth for GPU if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
                    
        except ImportError:
            print("TensorFlow not found. Please install tensorflow:")
            print("pip install tensorflow")
            return False
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        tf.random.set_seed(config.random_seed)
        
        # Create necessary directories
        print("\n1. Setting up directories...")
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_path, exist_ok=True)
        
        # Check if data exists
        print("\n2. Checking data availability...")
        if not os.path.exists(config.lightkurve_data_dir):
            print(f"Error: Light curve data directory not found: {config.lightkurve_data_dir}")
            print("Please ensure the data is in the correct location.")
            return False
            
        if not os.path.exists(config.feature_data_file):
            print(f"Error: Feature data file not found: {config.feature_data_file}")
            print("Please ensure the KOI data file is in the correct location.")
            return False
        
        # Load and preprocess data
        print("\n3. Loading and preprocessing data...")
        print(f"Light curve directory: {config.lightkurve_data_dir}")
        print(f"Feature file: {config.feature_data_file}")
        
        data_results = load_and_preprocess_data(
            lightcurve_dir=config.lightkurve_data_dir,
            feature_file=config.feature_data_file,
            time_series_length=config.time_series_length,
            test_size=config.test_split,
            val_size=config.validation_split,
            random_state=config.random_seed
        )
        
        (train_sequences, val_sequences, test_sequences,
         train_features, val_features, test_features,
         train_labels, val_labels, test_labels,
         feature_names, feature_scaler, split_info) = data_results
        
        print(f"\nData loading completed successfully!")
        print(f"Training set: {len(train_sequences)} samples")
        print(f"Validation set: {len(val_sequences)} samples")
        print(f"Test set: {len(test_sequences)} samples")
        print(f"Time series length: {train_sequences.shape[1]}")
        print(f"Number of features: {train_features.shape[1]}")
        print(f"Feature names: {feature_names}")
        
        # Update config with actual dimensions
        config.num_features = train_features.shape[1]
        config.time_series_length = train_sequences.shape[1]
        
        # Create and train model
        print("\n4. Creating and training model...")
        
        try:
            from models.dnn_model import (
                create_dual_input_dnn_model, train_model, evaluate_model,
                plot_training_history, plot_roc_curve
            )
            
            # Create model
            time_series_shape = (config.time_series_length,)
            feature_shape = (config.num_features,)
            
            model = create_dual_input_dnn_model(time_series_shape, feature_shape, config)
            
            print("\nModel architecture created:")
            model.summary()
            
            # Train model
            train_data = (train_sequences, train_features, train_labels)
            val_data = (val_sequences, val_features, val_labels)
            
            history = train_model(model, train_data, val_data, config)
            
            # Plot training history
            print("\n5. Plotting training history...")
            plot_path = os.path.join(config.log_dir, 'training_history.png')
            plot_training_history(history, save_path=plot_path)
            
            # Evaluate model
            print("\n6. Evaluating model...")
            test_data = (test_sequences, test_features, test_labels)
            results = evaluate_model(model, test_data, feature_names)
            
            # Plot ROC curve
            roc_path = os.path.join(config.log_dir, 'roc_curve.png')
            plot_roc_curve(test_labels, results['predicted_probs'], save_path=roc_path)
            
            # Save model in both formats
            print("\n7. Saving model in both .keras and .h5 formats...")
            
            # Save in .keras format (recommended)
            keras_path = config.model_save_path.replace('.h5', '.keras')
            model.save(keras_path)
            print(f"Model saved in .keras format: {keras_path}")
            
            # Save in .h5 format (for compatibility)
            model.save(config.model_save_path)
            print(f"Model saved in .h5 format: {config.model_save_path}")
            
            # Save additional artifacts
            print("\n8. Saving model artifacts...")
            
            # Save feature scaler and names
            import joblib
            scaler_path = os.path.join(os.path.dirname(config.model_save_path), 'feature_scaler.pkl')
            feature_names_path = os.path.join(os.path.dirname(config.model_save_path), 'feature_names.pkl')
            
            joblib.dump(feature_scaler, scaler_path)
            joblib.dump(feature_names, feature_names_path)
            
            # Save evaluation results
            results_path = os.path.join(config.log_dir, 'evaluation_results.txt')
            with open(results_path, 'w') as f:
                f.write("=== EXOPLANET CLASSIFICATION MODEL EVALUATION ===\n\n")
                f.write(f"Model Configuration:\n")
                f.write(f"  Time series length: {config.time_series_length}\n")
                f.write(f"  Number of features: {config.num_features}\n")
                f.write(f"  Training samples: {len(train_labels)}\n")
                f.write(f"  Validation samples: {len(val_labels)}\n")
                f.write(f"  Test samples: {len(test_labels)}\n\n")
                
                f.write(f"Performance Metrics:\n")
                f.write(f"  Test Accuracy: {results['test_accuracy']:.4f}\n")
                f.write(f"  Test Precision: {results['test_precision']:.4f}\n")
                f.write(f"  Test Recall: {results['test_recall']:.4f}\n")
                f.write(f"  AUC Score: {results['auc_score']:.4f}\n\n")
                
                f.write("Classification Report:\n")
                f.write(results['classification_report'])
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(results['confusion_matrix']))
                f.write(f"\n\nFeature Names Used:\n")
                for i, name in enumerate(feature_names):
                    f.write(f"  {i+1}. {name}\n")
            
            print("\n9. Creating comprehensive visualizations and analysis...")
            
            # Import visualization functions
            try:
                from visualization.visualize import create_comprehensive_analysis
                
                # Prepare data for comprehensive analysis
                y_pred = (results['predicted_probs'] > 0.5).astype(int).flatten()
                y_proba = results['predicted_probs'].flatten()
                test_data_with_predictions = (test_sequences, test_features, test_labels, y_pred, y_proba)
                
                # Get the split IDs from preprocessing
                train_ids = split_info['train_ids']
                val_ids = split_info['val_ids'] 
                test_ids = split_info['test_ids']
                
                # Get all IDs and labels
                all_ids = train_ids + val_ids + test_ids
                all_labels = np.concatenate([train_labels, val_labels, test_labels])
                
                # Create comprehensive analysis with automatic saving
                viz_dir = create_comprehensive_analysis(
                    history=history,
                    model=model,
                    test_data=test_data_with_predictions,
                    feature_names=feature_names,
                    lightcurve_dir=config.lightkurve_data_dir,
                    kepler_ids=all_ids,
                    labels=all_labels,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    save_base_dir='visualizations'
                )
                
                print(f"📊 Comprehensive analysis saved to: {viz_dir}")
                
            except Exception as e:
                print(f"Error creating comprehensive analysis: {e}")
                print("Continuing with basic explainability...")
            
            # Create model explainability report (if SHAP is available)
            print("\n10. Creating explainability report...")
            try:
                from visualization.explainability import create_explanation_report
                
                explanation_dir = os.path.join(config.log_dir, 'explanations')
                background_data = (train_sequences[:100], train_features[:100])  # Use subset for background
                
                explanation_results = create_explanation_report(
                    model, test_data, background_data, feature_names, 
                    config.time_series_length, explanation_dir
                )
                
                print("Explainability report created successfully!")
                
            except ImportError:
                print("SHAP not available. Skipping explainability report.")
                print("To enable explainability, install SHAP: pip install shap")
            except Exception as e:
                print(f"Error creating explainability report: {e}")
            
            # Final summary
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            # Create detailed model architecture documentation
            print("\n📝 Creating model architecture documentation...")
            create_model_architecture_documentation(
                model, feature_names, config, results, 
                os.path.join('models', 'model_architecture.md')
            )
            
            print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"Final AUC Score: {results['auc_score']:.4f}")
            print(f"\nModel saved to: {config.model_save_path.replace('.h5', '.keras')} (.keras format)")
            print(f"Model saved to: {config.model_save_path} (.h5 format)")
            print(f"Best checkpoint: {os.path.join(config.checkpoint_path, 'best_model.h5')}")
            print(f"Logs and plots: {config.log_dir}")
            print(f"Evaluation report: {results_path}")
            print(f"📖 Model architecture documentation: models/model_architecture.md")
            if 'viz_dir' in locals():
                print(f"📊 Comprehensive visualizations: {viz_dir}")
                print(f"📁 Data splits saved for reproducibility")
                print(f"🔍 Model explainability analysis included")
            
            return True
            
        except ImportError as e:
            print(f"Error importing model functions: {e}")
            print("Please ensure all dependencies are installed.")
            return False
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure the project structure is correct and dependencies are installed.")
        return False
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_model_architecture_documentation(model, feature_names, config, results, save_path):
    """
    Create detailed markdown documentation of the model architecture.
    """
    try:
        import tensorflow as tf
        from datetime import datetime
        
        # Ensure the models directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get model information
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Create markdown content
        markdown_content = f"""# Exoplanet Classification Model Architecture

## Overview
This document provides a comprehensive description of the dual-input deep neural network architecture designed for exoplanet candidate classification.

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Summary

- **Model Type:** Dual-Input Deep Neural Network (DNN)
- **Architecture:** Combined LSTM and Dense layers for time series and tabular data
- **Total Parameters:** {total_params:,}
- **Trainable Parameters:** {trainable_params:,}
- **Non-trainable Parameters:** {non_trainable_params:,}
- **Final Test Accuracy:** {results.get('test_accuracy', 'N/A'):.4f}
- **AUC Score:** {results.get('auc_score', 'N/A'):.4f}

## Architecture Details

### Input Specifications

#### 1. Time Series Input (Light Curves)
- **Shape:** `({config.time_series_length}, 1)`
- **Description:** Normalized light curve data representing flux measurements over time
- **Preprocessing:** 
  - Sequence padding/truncation to fixed length
  - Normalization using StandardScaler
  - Missing value handling

#### 2. Engineered Features Input
- **Shape:** `({len(feature_names) if feature_names else 'N/A'},)`
- **Features:** {len(feature_names) if feature_names else 'N/A'} engineered statistical features
- **Description:** Hand-crafted features extracted from light curves including:
  - Statistical moments (mean, std, skewness, kurtosis)
  - Frequency domain features
  - Transit-specific features
  - Variability metrics

### Network Architecture

"""

        # Add layer-by-layer description
        markdown_content += "### Layer Details\n\n"
        
        for i, layer in enumerate(model.layers):
            layer_name = layer.__class__.__name__
            layer_config = layer.get_config()
            layer_params = layer.count_params()
            
            try:
                output_shape = str(layer.output_shape)
            except:
                output_shape = "Variable"
            
            markdown_content += f"#### Layer {i+1}: {layer.name} ({layer_name})\n"
            markdown_content += f"- **Type:** {layer_name}\n"
            markdown_content += f"- **Output Shape:** {output_shape}\n"
            markdown_content += f"- **Parameters:** {layer_params:,}\n"
            
            # Add specific configuration details based on layer type
            if hasattr(layer, 'units') and layer.units:
                markdown_content += f"- **Units:** {layer.units}\n"
            if hasattr(layer, 'activation') and layer.activation:
                activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
                markdown_content += f"- **Activation:** {activation_name}\n"
            if hasattr(layer, 'rate') and layer.rate:
                markdown_content += f"- **Dropout Rate:** {layer.rate}\n"
            if hasattr(layer, 'return_sequences') and hasattr(layer, 'units'):
                markdown_content += f"- **Return Sequences:** {layer.return_sequences}\n"
            
            markdown_content += "\n"

        # Add training configuration
        markdown_content += f"""
### Training Configuration

- **Time Series Length:** {config.time_series_length}
- **Batch Size:** {config.batch_size}
- **Learning Rate:** {config.learning_rate}
- **Max Epochs:** {config.epochs}
- **Early Stopping Patience:** {config.early_stopping_patience}
- **Validation Split:** {config.validation_split}
- **Test Split:** {config.test_split}

### Model Compilation

- **Optimizer:** {model.optimizer.__class__.__name__ if hasattr(model, 'optimizer') else 'N/A'}
- **Loss Function:** {model.loss if hasattr(model, 'loss') else 'N/A'}
- **Metrics:** {', '.join([str(m) for m in model.metrics]) if hasattr(model, 'metrics') else 'N/A'}

### Performance Metrics

| Metric | Value |
|--------|--------|
| Test Accuracy | {results.get('test_accuracy', 'N/A'):.4f} |
| Test Precision | {results.get('test_precision', 'N/A'):.4f} |
| Test Recall | {results.get('test_recall', 'N/A'):.4f} |
| AUC Score | {results.get('auc_score', 'N/A'):.4f} |

## Model Architecture Rationale

### Dual-Input Design
The model employs a dual-input architecture to leverage both temporal patterns in light curves and statistical features:

1. **LSTM Branch:** Processes sequential light curve data to capture temporal dependencies and transit patterns
2. **Dense Branch:** Processes engineered features to capture statistical properties and domain-specific characteristics
3. **Fusion Layer:** Combines both representations for final classification

### Key Design Decisions

1. **LSTM for Time Series:** Captures long-term dependencies in light curve data, essential for detecting periodic transit signals
2. **Feature Engineering:** Incorporates domain knowledge through hand-crafted features that complement learned representations
3. **Dropout Regularization:** Prevents overfitting in the high-dimensional feature space
4. **Binary Classification:** Optimized for distinguishing between exoplanet candidates and false positives

## Data Pipeline

### Preprocessing Steps
1. Light curve normalization and cleaning
2. Sequence padding/truncation to fixed length
3. Feature engineering from raw time series
4. Train/validation/test splitting with stratification
5. Feature scaling using StandardScaler

### Augmentation Techniques
- Temporal shifts and scaling (if implemented)
- Noise injection for robustness
- Class balancing strategies

## Model Files

- **Keras Format:** `models/dual_input_dnn_model.keras` (Recommended)
- **H5 Format:** `models/dual_input_dnn_model.h5` (Compatibility)
- **Architecture Summary:** `models/model_architecture_summary.txt`
- **Training Logs:** `logs/training_history.png`

## Usage Instructions

```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('models/dual_input_dnn_model.keras')

# Make predictions
predictions = model.predict([time_series_data, engineered_features])
```

## References

- Kepler Space Telescope Data
- NASA Exoplanet Archive
- TensorFlow/Keras Documentation

---
*This documentation was automatically generated by the Exoplanet Classification Training Pipeline.*
"""

        # Write to file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Model architecture documentation saved to: {save_path}")
        
    except Exception as e:
        print(f"Error creating model documentation: {e}")
        import traceback
        traceback.print_exc()

def install_dependencies():
    """
    Install required dependencies.
    """
    print("Installing required dependencies...")
    
    dependencies = [
        "numpy",
        "pandas", 
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tensorflow",
        "shap",
        "scipy",
        "joblib"
    ]
    
    import subprocess
    import sys
    
    for package in dependencies:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exoplanet Classification Training Pipeline")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install required dependencies before training")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check data availability and configuration")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
        print("Dependencies installation completed.")
        if not args.check_only:
            print("\nStarting training...")
    
    if args.check_only:
        print("Performing data and configuration check...")
        # Add configuration check logic here
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data', 'raw', 'lightkurve_data')
        feature_file = os.path.join(current_dir, 'data', 'raw', 'KOI Selected 2000 Signals.csv')
        
        print(f"Checking data directory: {data_dir}")
        print(f"Directory exists: {os.path.exists(data_dir)}")
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            print(f"Number of light curve files: {len(csv_files)}")
        
        print(f"Checking feature file: {feature_file}")
        print(f"File exists: {os.path.exists(feature_file)}")
        
        sys.exit(0)
    
    success = main()
    if success:
        print("\n🎉 Training completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Training failed!")
        sys.exit(1)