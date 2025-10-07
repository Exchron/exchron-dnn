import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from .data_loader import get_available_kepler_ids, create_labels_from_disposition, load_single_lightcurve

def normalize_lightcurve(flux: np.ndarray, method: str = 'robust') -> np.ndarray:
    """
    Normalize light curve flux values.
    
    Parameters:
    -----------
    flux : np.ndarray
        Raw flux values
    method : str
        Normalization method ('robust', 'standard', 'minmax')
        
    Returns:
    --------
    np.ndarray
        Normalized flux values
    """
    if len(flux) == 0:
        return flux
    
    # Remove outliers first
    flux = remove_outliers(flux)
    
    if method == 'robust':
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        if mad > 0:
            flux = (flux - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
        else:
            flux = flux - median
    elif method == 'standard':
        mean = np.mean(flux)
        std = np.std(flux)
        if std > 0:
            flux = (flux - mean) / std
        else:
            flux = flux - mean
    elif method == 'minmax':
        min_val = np.min(flux)
        max_val = np.max(flux)
        if max_val > min_val:
            flux = (flux - min_val) / (max_val - min_val)
        
    return flux

def remove_outliers(flux: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Remove outliers from flux data using sigma clipping.
    
    Parameters:
    -----------
    flux : np.ndarray
        Flux values
    threshold : float
        Number of standard deviations for outlier detection
        
    Returns:
    --------
    np.ndarray
        Flux with outliers removed
    """
    if len(flux) == 0:
        return flux
    
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    
    if mad == 0:
        return flux
    
    modified_z_scores = 0.6745 * (flux - median) / mad
    mask = np.abs(modified_z_scores) < threshold
    
    return flux[mask]

def pad_or_truncate_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate a sequence to a target length.
    
    Parameters:
    -----------
    sequence : np.ndarray
        Input sequence
    target_length : int
        Target length
        
    Returns:
    --------
    np.ndarray
        Sequence of target length
    """
    if len(sequence) == target_length:
        return sequence
    elif len(sequence) > target_length:
        # Truncate from center
        start_idx = (len(sequence) - target_length) // 2
        return sequence[start_idx:start_idx + target_length]
    else:
        # Pad with zeros
        pad_length = target_length - len(sequence)
        pad_before = pad_length // 2
        pad_after = pad_length - pad_before
        return np.pad(sequence, (pad_before, pad_after), mode='constant', constant_values=0)

def process_lightcurve_file(file_path: str, target_length: int = 3000, normalize: bool = True) -> Optional[np.ndarray]:
    """
    Process a single light curve file.
    
    Parameters:
    -----------
    file_path : str
        Path to light curve file
    target_length : int
        Target length for the time series
    normalize : bool
        Whether to normalize the flux
        
    Returns:
    --------
    np.ndarray or None
        Processed flux sequence
    """
    flux = load_single_lightcurve(file_path)
    
    if flux is None or len(flux) == 0:
        return None
    
    # Remove outliers and normalize
    if normalize:
        flux = normalize_lightcurve(flux, method='robust')
    
    # Pad or truncate to target length
    flux = pad_or_truncate_sequence(flux, target_length)
    
    return flux

def extract_features_from_koi_data(feature_df: pd.DataFrame, kepler_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract numerical features from KOI dataset.
    
    Parameters:
    -----------
    feature_df : pd.DataFrame
        KOI feature DataFrame
    kepler_ids : List[str]
        List of Kepler IDs to extract features for
        
    Returns:
    --------
    tuple
        (feature_matrix, feature_names) - Features as numpy array and list of feature names
    """
    # Select numerical features that are commonly available and meaningful
    feature_columns = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_model_snr',
        'koi_impact', 'koi_sma', 'koi_incl', 'koi_steff', 'koi_slogg',
        'koi_srad', 'koi_smass', 'koi_kepmag'
    ]
    
    # Filter to only include columns that exist in the DataFrame
    available_features = [col for col in feature_columns if col in feature_df.columns]
    
    print(f"Using features: {available_features}")
    
    # Convert kepler_ids to int for matching
    kepler_ids_int = [int(kid) for kid in kepler_ids]
    
    # Create feature matrix maintaining exact order of kepler_ids
    feature_matrix = []
    for kid in kepler_ids_int:
        kid_data = feature_df[feature_df['kepid'] == kid]
        if len(kid_data) > 0:
            features = kid_data[available_features].iloc[0].values
        else:
            # If no data found, use zeros (will be imputed later)
            features = np.full(len(available_features), np.nan)
        feature_matrix.append(features)
    
    feature_matrix = np.array(feature_matrix)
    
    # Handle missing values by filling with median
    for i, col in enumerate(available_features):
        col_data = feature_matrix[:, i]
        mask = ~np.isnan(col_data)
        if np.any(mask):
            median_val = np.median(col_data[mask])
            feature_matrix[~mask, i] = median_val
        else:
            feature_matrix[:, i] = 0
    
    return feature_matrix, available_features

def create_train_test_split(kepler_ids: List[str], labels: np.ndarray, 
                          test_size: float = 0.2, val_size: float = 0.15, 
                          random_state: int = 42) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/validation/test splits for Kepler IDs and labels.
    
    Parameters:
    -----------
    kepler_ids : List[str]
        List of Kepler IDs
    labels : np.ndarray
        Corresponding labels
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set (from remaining after test split)
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        (train_ids, val_ids, test_ids, train_labels, val_labels, test_labels)
    """
    # First split: separate test set
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        kepler_ids, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate train and validation
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_val_ids, train_val_labels, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=train_val_labels
    )
    
    print(f"Data split:")
    print(f"  Train: {len(train_ids)} samples ({np.mean(train_labels):.2%} positive)")
    print(f"  Validation: {len(val_ids)} samples ({np.mean(val_labels):.2%} positive)")
    print(f"  Test: {len(test_ids)} samples ({np.mean(test_labels):.2%} positive)")
    
    return train_ids, val_ids, test_ids, train_labels, val_labels, test_labels

def save_split_info_to_csv(feature_df: pd.DataFrame, train_ids: List[str], val_ids: List[str], 
                          test_ids: List[str], train_labels: np.ndarray, val_labels: np.ndarray, 
                          test_labels: np.ndarray, output_dir: str = 'data/processed') -> None:
    """
    Save train/validation/test split information to CSV files.
    
    Parameters:
    -----------
    feature_df : pd.DataFrame
        Original feature DataFrame with all KOI data
    train_ids, val_ids, test_ids : List[str]
        Lists of Kepler IDs for each split
    train_labels, val_labels, test_labels : np.ndarray
        Labels for each split
    output_dir : str
        Directory to save CSV files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    def create_split_df(ids, labels, split_name):
        """Create DataFrame for a specific split"""
        # Convert IDs to integers for matching
        ids_int = [int(kid) for kid in ids]
        
        # Filter feature_df to only include the IDs in this split
        split_df = feature_df[feature_df['kepid'].isin(ids_int)].copy()
        
        # Remove duplicates if any exist, keeping the first occurrence
        split_df = split_df.drop_duplicates(subset=['kepid'], keep='first')
        
        # Create a mapping from kepid to row for reordering
        split_dict = split_df.set_index('kepid').to_dict('index')
        
        # Create ordered DataFrame matching the ids order
        ordered_rows = []
        valid_labels = []
        
        for i, kid in enumerate(ids_int):
            if kid in split_dict:
                row = split_dict[kid]
                row['kepid'] = kid  # Ensure kepid is included
                ordered_rows.append(row)
                if i < len(labels):
                    valid_labels.append(labels[i])
                else:
                    valid_labels.append(0)  # Default label if missing
        
        # Create the final DataFrame
        if ordered_rows:
            split_df = pd.DataFrame(ordered_rows)
            # Add split information
            split_df['split'] = split_name
            split_df['split_label'] = valid_labels
        else:
            # If no valid rows, create empty DataFrame with correct columns
            split_df = pd.DataFrame(columns=list(feature_df.columns) + ['split', 'split_label'])
        
        return split_df
    
    # Create DataFrames for each split
    train_df = create_split_df(train_ids, train_labels, 'train')
    val_df = create_split_df(val_ids, val_labels, 'validation')
    test_df = create_split_df(test_ids, test_labels, 'test')
    
    # Save individual split files
    train_path = os.path.join(output_dir, 'train_data.csv')
    val_path = os.path.join(output_dir, 'validation_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Also create a combined file with all splits
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_path = os.path.join(output_dir, 'data_splits_combined.csv')
    combined_df.to_csv(combined_path, index=False)
    
    print(f"\n📊 Split information saved:")
    print(f"   Training data: {train_path} ({len(train_df)} samples)")
    print(f"   Validation data: {val_path} ({len(val_df)} samples)")
    print(f"   Test data: {test_path} ({len(test_df)} samples)")
    print(f"   Combined splits: {combined_path} ({len(combined_df)} samples)")
    
    # Print summary statistics
    print(f"\n📈 Split summary:")
    print(f"   Train: {len(train_df)} samples ({np.mean(train_labels):.1%} candidates)")
    print(f"   Validation: {len(val_df)} samples ({np.mean(val_labels):.1%} candidates)")
    print(f"   Test: {len(test_df)} samples ({np.mean(test_labels):.1%} candidates)")

def load_and_preprocess_data(lightcurve_dir: str, feature_file: str, 
                           time_series_length: int = 3000, test_size: float = 0.2, 
                           val_size: float = 0.15, random_state: int = 42) -> Tuple:
    """
    Complete data loading and preprocessing pipeline.
    
    Parameters:
    -----------
    lightcurve_dir : str
        Directory containing light curve files
    feature_file : str
        Path to KOI feature file
    time_series_length : int
        Target length for time series
    test_size : float
        Test set proportion
    val_size : float
        Validation set proportion
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        Processed data ready for training
    """
    print("Loading data...")
    
    # Get available Kepler IDs and feature data
    available_ids, feature_df = get_available_kepler_ids(lightcurve_dir, feature_file)
    feature_df = create_labels_from_disposition(feature_df)
    
    # Filter feature data to available IDs
    kepler_ids_int = [int(kid) for kid in available_ids]
    filtered_feature_df = feature_df[feature_df['kepid'].isin(kepler_ids_int)].copy()
    
    # Get labels in same order as available_ids
    labels = []
    valid_ids = []
    
    for kid in available_ids:
        kid_int = int(kid)
        if kid_int in filtered_feature_df['kepid'].values:
            label = filtered_feature_df[filtered_feature_df['kepid'] == kid_int]['label'].iloc[0]
            labels.append(label)
            valid_ids.append(kid)
    
    labels = np.array(labels)
    
    print(f"Final dataset: {len(valid_ids)} objects with {np.mean(labels):.2%} candidates")
    
    # Create train/test splits
    train_ids, val_ids, test_ids, train_labels, val_labels, test_labels = create_train_test_split(
        valid_ids, labels, test_size, val_size, random_state
    )
    
    # Save split information to CSV files (without lightcurve data)
    save_split_info_to_csv(filtered_feature_df, train_ids, val_ids, test_ids, 
                          train_labels, val_labels, test_labels)
    
    # Process light curves
    print("Processing light curves...")
    
    def load_lightcurves_for_ids(id_list):
        sequences = []
        valid_sequences = []
        
        for kid in id_list:
            file_path = os.path.join(lightcurve_dir, f'kepler_{kid}_lightkurve.csv')
            sequence = process_lightcurve_file(file_path, time_series_length, normalize=True)
            
            if sequence is not None:
                sequences.append(sequence)
                valid_sequences.append(True)
            else:
                # Create zero sequence for missing data
                sequences.append(np.zeros(time_series_length))
                valid_sequences.append(False)
                print(f"Warning: Could not load light curve for {kid}, using zeros")
        
        return np.array(sequences)
    
    train_sequences = load_lightcurves_for_ids(train_ids)
    val_sequences = load_lightcurves_for_ids(val_ids)
    test_sequences = load_lightcurves_for_ids(test_ids)
    
    # Extract features
    print("Extracting features...")
    train_features, feature_names = extract_features_from_koi_data(filtered_feature_df, train_ids)
    val_features, _ = extract_features_from_koi_data(filtered_feature_df, val_ids)
    test_features, _ = extract_features_from_koi_data(filtered_feature_df, test_ids)
    
    # Normalize features
    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(train_features)
    val_features = feature_scaler.transform(val_features)
    test_features = feature_scaler.transform(test_features)
    
    print("Data preprocessing completed!")
    
    # Save the split information for later use
    split_info = {
        'train_ids': train_ids,
        'val_ids': val_ids, 
        'test_ids': test_ids
    }
    
    return (train_sequences, val_sequences, test_sequences,
            train_features, val_features, test_features,
            train_labels, val_labels, test_labels,
            feature_names, feature_scaler, split_info)

def main():
    """Test the preprocessing pipeline"""
    lightcurve_dir = os.path.join('data', 'raw', 'lightkurve_data')
    feature_file = os.path.join('data', 'raw', 'KOI Selected 2000 Signals.csv')
    
    try:
        results = load_and_preprocess_data(lightcurve_dir, feature_file, 
                                         time_series_length=3000, test_size=0.2, val_size=0.15)
        
        (train_sequences, val_sequences, test_sequences,
         train_features, val_features, test_features,
         train_labels, val_labels, test_labels,
         feature_names, feature_scaler) = results
        
        print(f"\nFinal shapes:")
        print(f"Train sequences: {train_sequences.shape}")
        print(f"Train features: {train_features.shape}")
        print(f"Train labels: {train_labels.shape}")
        print(f"Feature names: {feature_names}")
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()