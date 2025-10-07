#!/usr/bin/env python3
"""
Training script with modifications for softmax output.
This script handles the conversion from binary labels to categorical format
required for the new softmax output layer.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import project modules
from config import config
from data.preprocessing import load_and_preprocess_data
from models.dnn_model import create_dual_input_dnn_model, train_model, evaluate_model
from models.enhanced_dnn_model import create_enhanced_dual_input_model

def convert_labels_for_softmax(labels):
    """
    Convert binary labels (0, 1) to the format expected by sparse_categorical_crossentropy.
    
    Parameters:
    -----------
    labels : np.ndarray
        Binary labels (0 for non-candidate, 1 for candidate)
        
    Returns:
    --------
    np.ndarray: Labels in the correct format for sparse_categorical_crossentropy
    """
    # For sparse_categorical_crossentropy, labels should remain as integers
    # 0 = non-candidate (false positive)
    # 1 = candidate
    return labels.astype(np.int32)

def prepare_data_for_training():
    """
    Prepare data for training with the modified model.
    
    Returns:
    --------
    tuple: (train_data, val_data, test_data, feature_names, feature_scaler)
    """
    print("Loading and preprocessing data...")
    
    # Define paths
    lightcurve_dir = os.path.join(config.raw_data_dir, "lightkurve_data")
    feature_file = config.feature_data_file
    
    # Load and preprocess data using the existing pipeline
    results = load_and_preprocess_data(
        lightcurve_dir=lightcurve_dir,
        feature_file=feature_file,
        time_series_length=config.time_series_length,
        test_size=config.test_split,
        val_size=config.validation_split,
        random_state=config.random_seed
    )
    
    (train_sequences, val_sequences, test_sequences,
     train_features, val_features, test_features,
     train_labels, val_labels, test_labels,
     feature_names, feature_scaler, split_info) = results
    
    # Convert labels for softmax output
    train_labels = convert_labels_for_softmax(train_labels)
    val_labels = convert_labels_for_softmax(val_labels)
    test_labels = convert_labels_for_softmax(test_labels)
    
    print(f"\nData shapes after preprocessing:")
    print(f"  Train sequences: {train_sequences.shape}")
    print(f"  Train features: {train_features.shape}")
    print(f"  Train labels: {train_labels.shape}")
    print(f"  Label distribution: {np.bincount(train_labels)}")
    
    # Prepare data tuples
    train_data = (train_sequences, train_features, train_labels)
    val_data = (val_sequences, val_features, val_labels)
    test_data = (test_sequences, test_features, test_labels)
    
    return train_data, val_data, test_data, feature_names, feature_scaler

def save_preprocessing_artifacts(feature_scaler, feature_names):
    """
    Save preprocessing artifacts for later use.
    
    Parameters:
    -----------
    feature_scaler : sklearn.preprocessing.StandardScaler
        Fitted feature scaler
    feature_names : list
        List of feature names
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save feature scaler
    scaler_path = os.path.join("models", "feature_scaler.pkl")
    joblib.dump(feature_scaler, scaler_path)
    print(f"Feature scaler saved to: {scaler_path}")
    
    # Save feature names
    feature_names_path = os.path.join("models", "feature_names.pkl")
    joblib.dump(feature_names, feature_names_path)
    print(f"Feature names saved to: {feature_names_path}")

def main():
    """
    Main training function.
    """
    print("="*60)
    print("EXOPLANET CLASSIFICATION MODEL TRAINING")
    print("Modified for Softmax Output (Two-Class Probabilities)")
    print("="*60)
    
    try:
        # Set random seeds for reproducibility
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Prepare data
        train_data, val_data, test_data, feature_names, feature_scaler = prepare_data_for_training()
        
        # Get input shapes
        sequences_train, features_train, _ = train_data
        time_series_shape = (sequences_train.shape[1],)
        feature_shape = (features_train.shape[1],)
        
        print(f"\nModel input shapes:")
        print(f"  Time series: {time_series_shape}")
        print(f"  Features: {feature_shape}")
        
        # Create model
        if config.use_enhanced_model:
            print("Creating enhanced dual-input model...")
            model = create_enhanced_dual_input_model(time_series_shape, feature_shape)
        else:
            print("Creating standard dual-input model...")
            model = create_dual_input_dnn_model(time_series_shape, feature_shape, config)
        
        print(f"\nModel summary:")
        model.summary()
        print(f"Total parameters: {model.count_params():,}")
        
        # Train model
        print(f"\nStarting training...")
        history = train_model(model, train_data, val_data, config)
        
        # Evaluate model
        print(f"\nEvaluating model...")
        results = evaluate_model(model, test_data, feature_names)
        
        # Save preprocessing artifacts
        save_preprocessing_artifacts(feature_scaler, feature_names)
        
        # Save evaluation results
        results_text = f"""=== EXOPLANET CLASSIFICATION MODEL EVALUATION ===
Modified Model with Softmax Output (Two-Class Probabilities)

Model Configuration:
  Time series length: {config.time_series_length}
  Number of features: {len(feature_names)}
  Training samples: {len(train_data[2])}
  Validation samples: {len(val_data[2])}
  Test samples: {len(test_data[2])}

Performance Metrics:
  Test Accuracy: {results['test_accuracy']:.4f}
  Test Precision: {results['test_precision']:.4f}
  Test Recall: {results['test_recall']:.4f}
  AUC Score: {results['auc_score']:.4f}

Classification Report:
{results['classification_report']}

Confusion Matrix:
{results['confusion_matrix']}

Feature Names Used:
"""
        
        for i, feature_name in enumerate(feature_names, 1):
            results_text += f"  {i}. {feature_name}\n"
        
        # Add information about the new output format
        results_text += f"""

New Output Format:
  The model now outputs probabilities for both classes:
  - candidate_probability: Probability that the signal is a true exoplanet candidate
  - non_candidate_probability: Probability that the signal is a false positive
  - Both probabilities sum to 1.0

Example output format:
  "candidate_probability": 0.7389113903045654,
  "non_candidate_probability": 0.26108860969543457

"""
        
        # Save results
        os.makedirs("logs", exist_ok=True)
        results_path = os.path.join("logs", "evaluation_results.txt")
        with open(results_path, 'w') as f:
            f.write(results_text)
        
        print(f"\nEvaluation results saved to: {results_path}")
        print(f"Model saved to: {config.model_save_path}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("The model now outputs probabilities for both classes:")
        print('  "candidate_probability": P(exoplanet candidate)')
        print('  "non_candidate_probability": P(false positive)')
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Training completed successfully!")
        print("You can now use the prediction API or test scripts.")
    else:
        print("\n❌ Training failed!")