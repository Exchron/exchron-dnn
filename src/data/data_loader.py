import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

def load_lightcurve_data(lightcurve_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load light curve data from CSV files in the specified directory.

    Parameters:
    -----------
    lightcurve_dir : str
        Directory containing light curve CSV files.

    Returns:
    --------
    dict
        A dictionary where keys are Kepler IDs and values are DataFrames containing the light curve data.
    """
    lightcurve_data = {}
    
    if not os.path.exists(lightcurve_dir):
        raise FileNotFoundError(f"Directory {lightcurve_dir} not found")
    
    csv_files = [f for f in os.listdir(lightcurve_dir) if f.endswith('.csv') and f.startswith('kepler_')]
    
    print(f"Found {len(csv_files)} light curve files")
    
    for filename in csv_files:
        try:
            # Extract Kepler ID from filename (format: kepler_XXXXXXX_lightkurve.csv)
            kepler_id = filename.split('_')[1]
            file_path = os.path.join(lightcurve_dir, filename)
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Basic validation
            if len(df) == 0:
                print(f"Warning: Empty file {filename}")
                continue
                
            lightcurve_data[kepler_id] = df
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return lightcurve_data

def load_feature_data(feature_file: str) -> pd.DataFrame:
    """
    Load feature data from the KOI CSV file.

    Parameters:
    -----------
    feature_file : str
        Path to the CSV file containing KOI features.

    Returns:
    --------
    DataFrame
        A DataFrame containing the features with proper column names.
    """
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file {feature_file} not found")
    
    try:
        # Read the CSV file, skipping comment lines
        df = pd.read_csv(feature_file, comment='#')
        
        print(f"Loaded feature data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading feature file: {e}")
        raise

def get_available_kepler_ids(lightcurve_dir: str, feature_file: str) -> Tuple[List[str], pd.DataFrame]:
    """
    Get the intersection of Kepler IDs available in both light curve data and feature data.
    
    Parameters:
    -----------
    lightcurve_dir : str
        Directory containing light curve CSV files.
    feature_file : str
        Path to the KOI feature file.
        
    Returns:
    --------
    tuple
        (available_kepler_ids, feature_df) - List of available Kepler IDs and the feature DataFrame
    """
    # Load feature data
    feature_df = load_feature_data(feature_file)
    
    # Get available light curve files
    lightcurve_files = [f for f in os.listdir(lightcurve_dir) 
                       if f.endswith('.csv') and f.startswith('kepler_')]
    
    # Extract Kepler IDs from filenames
    lightcurve_kepler_ids = set()
    for filename in lightcurve_files:
        try:
            kepler_id = filename.split('_')[1]
            lightcurve_kepler_ids.add(kepler_id)
        except IndexError:
            continue
    
    # Get Kepler IDs from feature data
    feature_kepler_ids = set(feature_df['kepid'].astype(str))
    
    # Find intersection
    available_kepler_ids = list(lightcurve_kepler_ids.intersection(feature_kepler_ids))
    
    print(f"Light curve files: {len(lightcurve_kepler_ids)}")
    print(f"Feature data entries: {len(feature_kepler_ids)}")
    print(f"Available for training: {len(available_kepler_ids)}")
    
    return available_kepler_ids, feature_df

def create_labels_from_disposition(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary labels from KOI disposition.
    
    Parameters:
    -----------
    feature_df : pd.DataFrame
        Feature DataFrame with 'koi_disposition' column
        
    Returns:
    --------
    pd.DataFrame
        Feature DataFrame with added 'label' column (1 for CANDIDATE, 0 for FALSE POSITIVE)
    """
    feature_df = feature_df.copy()
    
    # Create binary labels: 1 for CANDIDATE (potential exoplanet), 0 for FALSE POSITIVE
    feature_df['label'] = (feature_df['koi_disposition'] == 'CANDIDATE').astype(int)
    
    print(f"Label distribution:")
    print(feature_df['label'].value_counts())
    print(f"Percentage of candidates: {feature_df['label'].mean():.2%}")
    
    return feature_df

def load_single_lightcurve(file_path: str) -> Optional[np.ndarray]:
    """
    Load a single light curve file and extract flux values.
    
    Parameters:
    -----------
    file_path : str
        Path to the light curve CSV file
        
    Returns:
    --------
    np.ndarray or None
        Flux values as numpy array, or None if loading fails
    """
    try:
        df = pd.read_csv(file_path)
        
        # Use PDCSAP flux if available, otherwise use SAP flux
        if 'pdcsap_flux' in df.columns:
            flux = df['pdcsap_flux'].values
        elif 'sap_flux' in df.columns:
            flux = df['sap_flux'].values
        elif 'flux' in df.columns:
            flux = df['flux'].values
        else:
            print(f"No flux column found in {file_path}")
            return None
            
        # Remove NaN values
        flux = flux[~np.isnan(flux)]
        
        if len(flux) == 0:
            print(f"No valid flux data in {file_path}")
            return None
            
        return flux
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    # Define paths based on project structure
    lightcurve_dir = os.path.join('data', 'raw', 'lightkurve_data')
    feature_file = os.path.join('data', 'raw', 'KOI Selected 2000 Signals.csv')
    
    try:
        # Get available Kepler IDs
        available_ids, feature_df = get_available_kepler_ids(lightcurve_dir, feature_file)
        
        # Create labels
        feature_df = create_labels_from_disposition(feature_df)
        
        print(f"\nSuccessfully processed data for {len(available_ids)} objects")
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()