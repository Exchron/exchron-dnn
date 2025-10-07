"""
Main training script for the dual-input DNN exoplanet classification model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import config
from data.preprocessing import load_and_preprocess_data
from models.dnn_model import (
    create_dual_input_dnn_model, train_model, evaluate_model,
    plot_training_history, plot_roc_curve
)

def main():
    """
    Main training pipeline for the exoplanet classification model.
    """
    print("=== Exoplanet Classification DNN Training ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    try:
        # Load and preprocess data
        print("\n1. Loading and preprocessing data...")
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
         feature_names, feature_scaler) = data_results
        
        print(f"\nData shapes:")
        print(f"  Train: sequences {train_sequences.shape}, features {train_features.shape}")
        print(f"  Validation: sequences {val_sequences.shape}, features {val_features.shape}")
        print(f"  Test: sequences {test_sequences.shape}, features {test_features.shape}")
        
        # Update config with actual feature count
        config.num_features = train_features.shape[1]
        
        # Create model
        print("\n2. Creating dual-input DNN model...")
        time_series_shape = (config.time_series_length,)
        feature_shape = (config.num_features,)
        
        model = create_dual_input_dnn_model(time_series_shape, feature_shape, config)
        
        print("\nModel architecture:")
        model.summary()
        
        # Train model
        print("\n3. Training model...")
        train_data = (train_sequences, train_features, train_labels)
        val_data = (val_sequences, val_features, val_labels)
        
        history = train_model(model, train_data, val_data, config)
        
        # Plot training history
        print("\n4. Plotting training history...")
        plot_path = os.path.join(config.log_dir, 'training_history.png')
        plot_training_history(history, save_path=plot_path)
        
        # Evaluate model
        print("\n5. Evaluating model...")
        test_data = (test_sequences, test_features, test_labels)
        results = evaluate_model(model, test_data, feature_names)
        
        # Plot ROC curve
        roc_path = os.path.join(config.log_dir, 'roc_curve.png')
        plot_roc_curve(test_labels, results['predicted_probs'], save_path=roc_path)
        
        # Save feature names and scaler
        import joblib
        scaler_path = os.path.join(os.path.dirname(config.model_save_path), 'feature_scaler.pkl')
        feature_names_path = os.path.join(os.path.dirname(config.model_save_path), 'feature_names.pkl')
        
        joblib.dump(feature_scaler, scaler_path)
        joblib.dump(feature_names, feature_names_path)
        
        print(f"\nFeature scaler saved to: {scaler_path}")
        print(f"Feature names saved to: {feature_names_path}")
        
        # Save evaluation results
        results_path = os.path.join(config.log_dir, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write("=== Exoplanet Classification Model Evaluation ===\n\n")
            f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
            f.write(f"Test Precision: {results['test_precision']:.4f}\n")
            f.write(f"Test Recall: {results['test_recall']:.4f}\n")
            f.write(f"AUC Score: {results['auc_score']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(results['confusion_matrix']))
            f.write(f"\n\nFeature Names: {feature_names}")
        
        print(f"Evaluation results saved to: {results_path}")
        
        print("\n=== Training Complete ===")
        print(f"Final model saved to: {config.model_save_path}")
        print(f"Best model checkpoint: {os.path.join(config.checkpoint_path, 'best_model.h5')}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed!")
        sys.exit(1)