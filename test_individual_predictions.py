#!/usr/bin/env python3
"""
Test individual predictions on the first 5 records of the test set.
This script loads the trained .keras model and makes predictions on individual samples.
Modified to output probabilities for both candidate and non-candidate classes.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def load_model_and_artifacts():
    """
    Load the trained model and associated artifacts.
    
    Returns:
    --------
    tuple: (model, feature_scaler, feature_names)
    """
    print("Loading model and artifacts...")
    
    # Load the .keras model
    model_path = os.path.join("models", "dual_input_dnn_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")
    
    # Load feature scaler
    scaler_path = os.path.join("models", "feature_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")
    
    feature_scaler = joblib.load(scaler_path)
    print(f"✓ Feature scaler loaded from: {scaler_path}")
    
    # Load feature names
    feature_names_path = os.path.join("models", "feature_names.pkl")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names not found: {feature_names_path}")
    
    feature_names = joblib.load(feature_names_path)
    print(f"✓ Feature names loaded from: {feature_names_path}")
    print(f"Features: {feature_names}")
    
    return model, feature_scaler, feature_names

def load_test_data():
    """
    Load the test dataset.
    
    Returns:
    --------
    pd.DataFrame: Test data
    """
    print("\nLoading test data...")
    
    test_data_path = os.path.join("data", "processed", "test_data.csv")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    
    test_df = pd.read_csv(test_data_path)
    print(f"✓ Test data loaded: {len(test_df)} samples")
    print(f"Columns: {list(test_df.columns)}")
    
    return test_df

def load_single_lightcurve(kepid, lightcurve_dir, time_series_length=3000):
    """
    Load and preprocess a single light curve.
    
    Parameters:
    -----------
    kepid : int
        Kepler ID
    lightcurve_dir : str
        Directory containing light curve files
    time_series_length : int
        Target length for time series
        
    Returns:
    --------
    np.ndarray: Preprocessed light curve
    """
    from data.preprocessing import normalize_lightcurve, pad_or_truncate_sequence
    
    # Construct filename
    filename = f"kepler_{kepid}_lightkurve.csv"
    filepath = os.path.join(lightcurve_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: Light curve file not found for KepID {kepid}: {filepath}")
        # Return a sequence of zeros if file not found
        return np.zeros(time_series_length)
    
    try:
        # Load the light curve data
        lc_df = pd.read_csv(filepath)
        
        # Get flux values (assuming column is named 'flux' or 'pdcsap_flux')
        if 'flux' in lc_df.columns:
            flux = lc_df['flux'].values
        elif 'pdcsap_flux' in lc_df.columns:
            flux = lc_df['pdcsap_flux'].values
        else:
            # Use the second column if flux column not found
            flux = lc_df.iloc[:, 1].values
        
        # Remove NaN values
        flux = flux[~np.isnan(flux)]
        
        if len(flux) == 0:
            return np.zeros(time_series_length)
        
        # Normalize the flux
        flux_normalized = normalize_lightcurve(flux, method='robust')
        
        # Pad or truncate to desired length
        flux_processed = pad_or_truncate_sequence(flux_normalized, time_series_length)
        
        return flux_processed
        
    except Exception as e:
        print(f"Error loading light curve for KepID {kepid}: {e}")
        return np.zeros(time_series_length)

def prepare_sample_data(row, feature_scaler, feature_names, lightcurve_dir, time_series_length=3000):
    """
    Prepare data for a single sample prediction.
    
    Parameters:
    -----------
    row : pd.Series
        Row from test dataset
    feature_scaler : sklearn scaler
        Fitted feature scaler
    feature_names : list
        List of feature names
    lightcurve_dir : str
        Directory containing light curve files
    time_series_length : int
        Expected time series length
        
    Returns:
    --------
    tuple: (time_series_data, feature_data)
    """
    kepid = row['kepid']
    
    # Load and preprocess light curve
    time_series_data = load_single_lightcurve(kepid, lightcurve_dir, time_series_length)
    
    # Use only the features that the model was trained on
    # Extract feature values in the exact order expected by the scaler
    feature_values = []
    for feature_name in feature_names:
        if feature_name in row.index:
            value = row[feature_name]
            if pd.isna(value):
                value = 0.0  # Replace NaN with 0
            feature_values.append(float(value))
        else:
            print(f"Warning: Feature '{feature_name}' not found in data, using 0.0")
            feature_values.append(0.0)
    
    feature_data = np.array(feature_values).reshape(1, -1)
    
    # Scale features using the fitted scaler
    feature_data_scaled = feature_scaler.transform(feature_data)
    
    # Reshape time series data
    time_series_data = time_series_data.reshape(1, -1)
    
    return time_series_data, feature_data_scaled

def make_prediction(model, time_series_data, feature_data):
    """
    Make prediction using the dual-input model.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    time_series_data : np.ndarray
        Time series input data
    feature_data : np.ndarray
        Feature input data
        
    Returns:
    --------
    dict: Dictionary with candidate and non-candidate probabilities
    """
    # Make prediction - now returns probabilities for both classes
    predictions = model.predict([time_series_data, feature_data], verbose=0)[0]
    
    non_candidate_prob = predictions[0]
    candidate_prob = predictions[1]
    prediction_class = 1 if candidate_prob > 0.5 else 0
    
    return {
        "non_candidate_probability": float(non_candidate_prob),
        "candidate_probability": float(candidate_prob),
        "prediction_class": prediction_class
    }

def display_prediction_results(sample_idx, row, prediction_result, actual_label):
    """
    Display prediction results for a single sample.
    
    Parameters:
    -----------
    sample_idx : int
        Sample index
    row : pd.Series
        Sample data
    prediction_result : dict
        Prediction result with probabilities
    actual_label : int
        Actual label
    """
    kepid = row['kepid']
    koi_disposition = row['koi_disposition']
    
    candidate_prob = prediction_result["candidate_probability"]
    non_candidate_prob = prediction_result["non_candidate_probability"]
    prediction_class = prediction_result["prediction_class"]
    
    print(f"\n{'='*60}")
    print(f"SAMPLE {sample_idx + 1}")
    print(f"{'='*60}")
    print(f"Kepler ID: {kepid}")
    print(f"KOI Disposition: {koi_disposition}")
    print(f"Actual Label: {'CANDIDATE' if actual_label == 1 else 'FALSE POSITIVE'}")
    print(f"\nModel Prediction:")
    print(f'  "candidate_probability": {candidate_prob:.13f},')
    print(f'  "non_candidate_probability": {non_candidate_prob:.13f}')
    print(f"  Class: {'CANDIDATE' if prediction_class == 1 else 'FALSE POSITIVE'}")
    print(f"  Confidence: {max(candidate_prob, non_candidate_prob):.6f}")
    
    # Determine if prediction is correct
    is_correct = prediction_class == actual_label
    status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    print(f"\nPrediction Status: {status}")
    
    # Show key features
    print(f"\nKey Features:")
    if not pd.isna(row.get('koi_period')):
        print(f"  Period: {row['koi_period']:.3f} days")
    if not pd.isna(row.get('koi_depth')):
        print(f"  Transit Depth: {row['koi_depth']:.1f} ppm")
    if not pd.isna(row.get('koi_duration')):
        print(f"  Transit Duration: {row['koi_duration']:.3f} hours")
    if not pd.isna(row.get('koi_model_snr')):
        print(f"  Signal-to-Noise: {row['koi_model_snr']:.1f}")

def main():
    """
    Main function to test individual predictions.
    """
    print("="*60)
    print("INDIVIDUAL PREDICTION TESTING")
    print("Testing First 5 Records from Test Set")
    print("Using Trained .keras Model with Softmax Output")
    print("="*60)
    
    try:
        # Load model and artifacts
        model, feature_scaler, feature_names = load_model_and_artifacts()
        
        # Display model information
        print(f"\nModel Information:")
        print(f"  Input shapes: {[input_layer.shape for input_layer in model.inputs]}")
        print(f"  Output shape: {model.output.shape}")
        print(f"  Total parameters: {model.count_params():,}")
        
        # Load test data
        test_df = load_test_data()
        
        # Get first 5 records
        first_5_records = test_df.head(5)
        
        # Light curve directory
        lightcurve_dir = os.path.join("data", "raw", "lightkurve_data")
        
        print(f"\nTesting {len(first_5_records)} samples...")
        print(f"Light curve directory: {lightcurve_dir}")
        
        # Test each record
        results = []
        for idx, (_, row) in enumerate(first_5_records.iterrows()):
            print(f"\nProcessing sample {idx + 1}...")
            
            try:
                # Prepare data for this sample
                time_series_data, feature_data = prepare_sample_data(
                    row, feature_scaler, feature_names, lightcurve_dir
                )
                
                # Make prediction
                prediction_result = make_prediction(
                    model, time_series_data, feature_data
                )
                
                # Get actual label
                actual_label = row['label']
                
                # Display results
                display_prediction_results(
                    idx, row, prediction_result, actual_label
                )
                
                # Store results
                results.append({
                    'sample_idx': idx + 1,
                    'kepid': row['kepid'],
                    'actual_label': actual_label,
                    'candidate_probability': prediction_result["candidate_probability"],
                    'non_candidate_probability': prediction_result["non_candidate_probability"],
                    'predicted_class': prediction_result["prediction_class"],
                    'is_correct': prediction_result["prediction_class"] == actual_label
                })
                
            except Exception as e:
                print(f"Error processing sample {idx + 1}: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary results
        print(f"\n{'='*60}")
        print("SUMMARY RESULTS")
        print(f"{'='*60}")
        
        correct_predictions = sum(1 for r in results if r['is_correct'])
        total_predictions = len(results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"Total Samples Tested: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print(f"\nDetailed Results:")
        for result in results:
            status = "✓" if result['is_correct'] else "✗"
            print(f"  Sample {result['sample_idx']} (KepID {result['kepid']}): "
                  f"Candidate={result['candidate_probability']:.4f}, "
                  f"Non-candidate={result['non_candidate_probability']:.4f}, "
                  f"Pred={'CANDIDATE' if result['predicted_class'] == 1 else 'FALSE POSITIVE'}, "
                  f"Actual={'CANDIDATE' if result['actual_label'] == 1 else 'FALSE POSITIVE'} {status}")
        
        # Save results to file
        results_df = pd.DataFrame(results)
        output_path = os.path.join("logs", "individual_predictions_test.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Also save in JSON format with the exact requested structure
        json_results = []
        for result in results:
            json_results.append({
                "sample_idx": result['sample_idx'],
                "kepid": result['kepid'],
                "candidate_probability": result['candidate_probability'],
                "non_candidate_probability": result['non_candidate_probability'],
                "actual_label": "CANDIDATE" if result['actual_label'] == 1 else "FALSE POSITIVE",
                "predicted_class": "CANDIDATE" if result['predicted_class'] == 1 else "FALSE POSITIVE"
            })
        
        json_output_path = os.path.join("logs", "individual_predictions_test.json")
        with open(json_output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"JSON results saved to: {json_output_path}")
        
        # Show example of the exact format requested
        if json_results:
            print(f"\nExample output format (Sample 1):")
            example = json_results[0]
            print(f'  "candidate_probability": {example["candidate_probability"]:.13f},')
            print(f'  "non_candidate_probability": {example["non_candidate_probability"]:.13f}')
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Import required modules after path setup
    try:
        from data.preprocessing import normalize_lightcurve, pad_or_truncate_sequence
    except ImportError as e:
        print(f"Error importing preprocessing functions: {e}")
        print("Please ensure the project structure is correct.")
        sys.exit(1)
    
    success = main()
    if success:
        print(f"\n🎉 Individual prediction testing completed successfully!")
    else:
        print(f"\n❌ Individual prediction testing failed!")