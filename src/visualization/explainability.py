"""
SHAP-based explainability module for the exoplanet classification model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Any
import warnings
import os
warnings.filterwarnings('ignore')

def create_shap_explainer(model: tf.keras.Model, background_data: Tuple[np.ndarray, np.ndarray], 
                         max_evals: int = 100) -> shap.Explainer:
    """
    Create a SHAP explainer for the dual-input model.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained dual-input model
    background_data : tuple
        (background_sequences, background_features) for SHAP background
    max_evals : int
        Maximum number of evaluations for SHAP
        
    Returns:
    --------
    shap.Explainer
        SHAP explainer object
    """
    background_sequences, background_features = background_data
    
    # Create a wrapper function for the model
    def model_predict(data):
        if len(data.shape) == 2 and data.shape[1] == background_sequences.shape[1] + background_features.shape[1]:
            # Split the concatenated data back into sequences and features
            seq_data = data[:, :background_sequences.shape[1]]
            feat_data = data[:, background_sequences.shape[1]:]
            return model.predict([seq_data, feat_data])
        else:
            # Assume it's already in the correct format
            return model.predict(data)
    
    # Concatenate background data for SHAP
    background_combined = np.concatenate([background_sequences, background_features], axis=1)
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(model_predict, background_combined[:max_evals])
    
    return explainer

def get_shap_values(explainer: shap.Explainer, test_data: Tuple[np.ndarray, np.ndarray], 
                   num_samples: int = 50) -> np.ndarray:
    """
    Calculate SHAP values for test data.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    test_data : tuple
        (test_sequences, test_features) to explain
    num_samples : int
        Number of samples to explain
        
    Returns:
    --------
    np.ndarray
        SHAP values
    """
    test_sequences, test_features = test_data
    
    # Take a subset of test data
    indices = np.random.choice(len(test_sequences), min(num_samples, len(test_sequences)), replace=False)
    
    # Concatenate test data
    test_combined = np.concatenate([test_sequences[indices], test_features[indices]], axis=1)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(test_combined)
    
    return shap_values, indices

def plot_shap_summary(shap_values: np.ndarray, test_data: Tuple[np.ndarray, np.ndarray], 
                     feature_names: List[str], time_series_length: int, 
                     save_path: Optional[str] = None) -> None:
    """
    Plot SHAP summary plot.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    test_data : tuple
        Test data used for explanation
    feature_names : list
        Names of the features
    time_series_length : int
        Length of time series portion
    save_path : str, optional
        Path to save the plot
    """
    test_sequences, test_features = test_data
    
    # Combine test data
    test_combined = np.concatenate([test_sequences, test_features], axis=1)
    
    # Create feature names for the combined data
    time_series_names = [f'TS_{i}' for i in range(time_series_length)]
    all_feature_names = time_series_names + feature_names
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, test_combined, 
                     feature_names=all_feature_names, 
                     show=False, max_display=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    
    plt.show()

def plot_feature_importance_comparison(shap_values: np.ndarray, feature_names: List[str], 
                                     time_series_length: int, save_path: Optional[str] = None) -> None:
    """
    Compare importance between time series features and extracted features.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values
    feature_names : list
        Names of the extracted features
    time_series_length : int
        Length of time series portion
    save_path : str, optional
        Path to save the plot
    """
    # Calculate mean absolute SHAP values
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Split into time series and feature importance
    ts_importance = np.sum(mean_shap[:time_series_length])
    feature_importance = mean_shap[time_series_length:]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall importance comparison
    categories = ['Time Series', 'Extracted Features']
    importances = [ts_importance, np.sum(feature_importance)]
    
    ax1.bar(categories, importances, color=['skyblue', 'lightcoral'])
    ax1.set_title('Overall Feature Importance Comparison')
    ax1.set_ylabel('Mean |SHAP Value|')
    
    # Individual extracted feature importance
    if len(feature_importance) > 0:
        sorted_indices = np.argsort(feature_importance)[::-1]
        top_features = sorted_indices[:min(15, len(feature_importance))]
        
        ax2.barh(range(len(top_features)), feature_importance[top_features], color='lightcoral')
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels([feature_names[i] for i in top_features])
        ax2.set_title('Top Extracted Features Importance')
        ax2.set_xlabel('Mean |SHAP Value|')
        ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance comparison saved to {save_path}")
    
    plt.show()

def create_explanation_report(model: tf.keras.Model, test_data: Tuple[np.ndarray, np.ndarray], 
                            background_data: Tuple[np.ndarray, np.ndarray], 
                            feature_names: List[str], time_series_length: int, 
                            output_dir: str = 'explanations') -> Dict[str, Any]:
    """
    Create a comprehensive explanation report with automatic saving.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    test_data : tuple
        Test data to explain
    background_data : tuple
        Background data for SHAP
    feature_names : list
        Names of extracted features
    time_series_length : int
        Length of time series
    output_dir : str
        Directory to save explanation plots
        
    Returns:
    --------
    dict
        Dictionary containing explanation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Creating SHAP explainer...")
        explainer = create_shap_explainer(model, background_data, max_evals=50)
        
        print("Calculating SHAP values...")
        shap_values, indices = get_shap_values(explainer, test_data, num_samples=30)
        
        # Extract subset of test data used for explanation
        test_sequences, test_features = test_data
        subset_sequences = test_sequences[indices]
        subset_features = test_features[indices]
        subset_data = (subset_sequences, subset_features)
        
        print("Creating SHAP plots...")
        
        # Summary plot
        plot_shap_summary(shap_values, subset_data, feature_names, time_series_length,
                         save_path=os.path.join(output_dir, 'shap_summary.png'))
        
        # Feature importance comparison
        plot_feature_importance_comparison(shap_values, feature_names, time_series_length,
                                         save_path=os.path.join(output_dir, 'feature_importance.png'))
        
        print(f"Explanation report saved to {output_dir}")
        
        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'explained_indices': indices,
            'feature_names': feature_names
        }
    
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Continuing without SHAP analysis...")
        return {
            'shap_values': None,
            'explainer': None,
            'explained_indices': None,
            'feature_names': feature_names,
            'error': str(e)
        }

def load_and_preprocess_data(lightcurve_dir, feature_file):
    """Legacy function for backward compatibility"""
    # Load light curve data
    lightcurve_files = [f for f in os.listdir(lightcurve_dir) if f.endswith('.csv')]
    lightcurves = []
    
    for file in lightcurve_files:
        df = pd.read_csv(os.path.join(lightcurve_dir, file))
        lightcurves.append(df)
    
    # Load features
    features = pd.read_csv(feature_file, comment='#')
    
    # Combine light curves and features as needed
    combined_data = pd.concat(lightcurves, ignore_index=True)
    
    return combined_data, features

def explain_model_predictions(model, X, feature_names):
    """Legacy function for backward compatibility"""
    # Create SHAP explainer
    explainer = shap.KernelExplainer(model.predict, X)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # Plot summary
    shap.summary_plot(shap_values, X, feature_names=feature_names)

def plot_shap_values(shap_values, feature_names):
    """Legacy function for backward compatibility"""
    # Plot SHAP values for the first instance
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], feature_names=feature_names)

def main():
    """
    Test the explainability functions with dummy data.
    """
    print("Testing explainability module with dummy data...")
    
    # Create dummy data
    n_samples = 100
    time_series_length = 3000
    n_features = 12
    
    # Dummy model (for testing)
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Concatenate
    
    ts_input = Input(shape=(time_series_length,))
    feat_input = Input(shape=(n_features,))
    
    ts_processed = Dense(64, activation='relu')(ts_input)
    feat_processed = Dense(32, activation='relu')(feat_input)
    
    combined = Concatenate()([ts_processed, feat_processed])
    output = Dense(1, activation='sigmoid')(combined)
    
    dummy_model = Model(inputs=[ts_input, feat_input], outputs=output)
    dummy_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Dummy data
    test_sequences = np.random.randn(n_samples, time_series_length)
    test_features = np.random.randn(n_samples, n_features)
    background_sequences = np.random.randn(20, time_series_length)
    background_features = np.random.randn(20, n_features)
    
    test_data = (test_sequences, test_features)
    background_data = (background_sequences, background_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Test explanation report
    try:
        results = create_explanation_report(
            dummy_model, test_data, background_data, 
            feature_names, time_series_length, 'test_explanations'
        )
        print("Explainability test completed successfully!")
    except Exception as e:
        print(f"Error in explainability test: {e}")

if __name__ == "__main__":
    main()