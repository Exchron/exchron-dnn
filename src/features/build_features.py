"""
Feature engineering module for exoplanet classification.
This module contains functions to extract statistical and domain-specific features
from light curve time series data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, periodogram
from typing import Dict, List, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

def extract_statistical_features(flux: np.ndarray) -> Dict[str, float]:
    """
    Extract basic statistical features from light curve flux data.
    
    Parameters:
    -----------
    flux : np.ndarray
        Flux values from light curve
        
    Returns:
    --------
    dict
        Dictionary of statistical features
    """
    if len(flux) == 0 or np.all(np.isnan(flux)):
        return {f'stat_{key}': 0.0 for key in [
            'mean', 'std', 'median', 'mad', 'skewness', 'kurtosis',
            'min', 'max', 'range', 'q25', 'q75', 'iqr'
        ]}
    
    # Remove NaN values
    clean_flux = flux[~np.isnan(flux)]
    
    if len(clean_flux) == 0:
        return {f'stat_{key}': 0.0 for key in [
            'mean', 'std', 'median', 'mad', 'skewness', 'kurtosis',
            'min', 'max', 'range', 'q25', 'q75', 'iqr'
        ]}
    
    features = {}
    
    # Basic statistics
    features['stat_mean'] = np.mean(clean_flux)
    features['stat_std'] = np.std(clean_flux)
    features['stat_median'] = np.median(clean_flux)
    features['stat_mad'] = np.median(np.abs(clean_flux - features['stat_median']))
    
    # Shape statistics
    try:
        features['stat_skewness'] = stats.skew(clean_flux)
        features['stat_kurtosis'] = stats.kurtosis(clean_flux)
    except:
        features['stat_skewness'] = 0.0
        features['stat_kurtosis'] = 0.0
    
    # Range statistics
    features['stat_min'] = np.min(clean_flux)
    features['stat_max'] = np.max(clean_flux)
    features['stat_range'] = features['stat_max'] - features['stat_min']
    
    # Percentiles
    features['stat_q25'] = np.percentile(clean_flux, 25)
    features['stat_q75'] = np.percentile(clean_flux, 75)
    features['stat_iqr'] = features['stat_q75'] - features['stat_q25']
    
    return features

def extract_variability_features(flux: np.ndarray, time: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extract variability features from light curve data.
    
    Parameters:
    -----------
    flux : np.ndarray
        Flux values
    time : np.ndarray, optional
        Time values (if None, assumes uniform sampling)
        
    Returns:
    --------
    dict
        Dictionary of variability features
    """
    if len(flux) == 0:
        return {f'var_{key}': 0.0 for key in [
            'amplitude', 'beyond_1std', 'beyond_2std', 'flux_percentile_ratio',
            'percent_amplitude', 'percent_difference_flux_percentile'
        ]}
    
    clean_flux = flux[~np.isnan(flux)]
    
    if len(clean_flux) < 3:
        return {f'var_{key}': 0.0 for key in [
            'amplitude', 'beyond_1std', 'beyond_2std', 'flux_percentile_ratio',
            'percent_amplitude', 'percent_difference_flux_percentile'
        ]}
    
    features = {}
    
    mean_flux = np.mean(clean_flux)
    std_flux = np.std(clean_flux)
    median_flux = np.median(clean_flux)
    
    # Amplitude features
    features['var_amplitude'] = (np.max(clean_flux) - np.min(clean_flux)) / 2.0
    
    if mean_flux != 0:
        features['var_percent_amplitude'] = 100 * features['var_amplitude'] / mean_flux
    else:
        features['var_percent_amplitude'] = 0.0
    
    # Flux excursion features
    if std_flux > 0:
        features['var_beyond_1std'] = np.sum(np.abs(clean_flux - mean_flux) > std_flux) / len(clean_flux)
        features['var_beyond_2std'] = np.sum(np.abs(clean_flux - mean_flux) > 2*std_flux) / len(clean_flux)
    else:
        features['var_beyond_1std'] = 0.0
        features['var_beyond_2std'] = 0.0
    
    # Percentile features
    flux_95 = np.percentile(clean_flux, 95)
    flux_5 = np.percentile(clean_flux, 5)
    
    if median_flux != 0:
        features['var_flux_percentile_ratio'] = (flux_95 - flux_5) / median_flux
        features['var_percent_difference_flux_percentile'] = 100 * (flux_95 - flux_5) / median_flux
    else:
        features['var_flux_percentile_ratio'] = 0.0
        features['var_percent_difference_flux_percentile'] = 0.0
    
    return features

def extract_frequency_features(flux: np.ndarray, time: Optional[np.ndarray] = None, 
                             sample_rate: float = 1.0) -> Dict[str, float]:
    """
    Extract frequency domain features from light curve data.
    
    Parameters:
    -----------
    flux : np.ndarray
        Flux values
    time : np.ndarray, optional
        Time values
    sample_rate : float
        Sampling rate (if time is None)
        
    Returns:
    --------
    dict
        Dictionary of frequency features
    """
    if len(flux) < 10:
        return {f'freq_{key}': 0.0 for key in [
            'dominant_freq', 'dominant_power', 'spectral_centroid', 
            'spectral_rolloff', 'spectral_flux'
        ]}
    
    clean_flux = flux[~np.isnan(flux)]
    
    if len(clean_flux) < 10:
        return {f'freq_{key}': 0.0 for key in [
            'dominant_freq', 'dominant_power', 'spectral_centroid', 
            'spectral_rolloff', 'spectral_flux'
        ]}
    
    features = {}
    
    try:
        # Compute periodogram
        frequencies, power = periodogram(clean_flux, fs=sample_rate)
        
        # Remove DC component
        if len(frequencies) > 1:
            frequencies = frequencies[1:]
            power = power[1:]
        
        if len(power) == 0:
            return {f'freq_{key}': 0.0 for key in [
                'dominant_freq', 'dominant_power', 'spectral_centroid', 
                'spectral_rolloff', 'spectral_flux'
            ]}
        
        # Dominant frequency and power
        max_power_idx = np.argmax(power)
        features['freq_dominant_freq'] = frequencies[max_power_idx]
        features['freq_dominant_power'] = power[max_power_idx]
        
        # Spectral centroid
        if np.sum(power) > 0:
            features['freq_spectral_centroid'] = np.sum(frequencies * power) / np.sum(power)
        else:
            features['freq_spectral_centroid'] = 0.0
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum_power = np.cumsum(power)
        total_power = cumsum_power[-1]
        
        if total_power > 0:
            rolloff_idx = np.where(cumsum_power >= 0.85 * total_power)[0]
            if len(rolloff_idx) > 0:
                features['freq_spectral_rolloff'] = frequencies[rolloff_idx[0]]
            else:
                features['freq_spectral_rolloff'] = frequencies[-1]
        else:
            features['freq_spectral_rolloff'] = 0.0
        
        # Spectral flux (measure of how quickly the power spectrum changes)
        if len(power) > 1:
            spectral_diff = np.diff(power)
            features['freq_spectral_flux'] = np.sum(spectral_diff ** 2)
        else:
            features['freq_spectral_flux'] = 0.0
            
    except Exception as e:
        # If frequency analysis fails, return zeros
        features = {f'freq_{key}': 0.0 for key in [
            'dominant_freq', 'dominant_power', 'spectral_centroid', 
            'spectral_rolloff', 'spectral_flux'
        ]}
    
    return features

def extract_transit_features(flux: np.ndarray, time: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extract features related to potential transit events.
    
    Parameters:
    -----------
    flux : np.ndarray
        Flux values
    time : np.ndarray, optional
        Time values
        
    Returns:
    --------
    dict
        Dictionary of transit-related features
    """
    if len(flux) < 10:
        return {f'transit_{key}': 0.0 for key in [
            'num_dips', 'deepest_dip', 'dip_duration_ratio', 'periodicity_score'
        ]}
    
    clean_flux = flux[~np.isnan(flux)]
    
    if len(clean_flux) < 10:
        return {f'transit_{key}': 0.0 for key in [
            'num_dips', 'deepest_dip', 'dip_duration_ratio', 'periodicity_score'
        ]}
    
    features = {}
    
    # Normalize flux to median
    median_flux = np.median(clean_flux)
    if median_flux != 0:
        normalized_flux = clean_flux / median_flux
    else:
        normalized_flux = clean_flux
    
    # Find dips (negative peaks in inverted signal)
    inverted_flux = -normalized_flux
    
    try:
        # Find peaks in inverted signal (dips in original)
        height_threshold = np.std(normalized_flux)
        peaks, properties = find_peaks(inverted_flux, height=height_threshold, 
                                     distance=len(clean_flux)//20)  # Minimum distance between dips
        
        features['transit_num_dips'] = len(peaks)
        
        if len(peaks) > 0:
            # Deepest dip
            dip_depths = inverted_flux[peaks]
            features['transit_deepest_dip'] = np.max(dip_depths)
            
            # Estimate dip duration ratio
            if 'widths' in properties:
                avg_width = np.mean(properties['widths'])
                features['transit_dip_duration_ratio'] = avg_width / len(clean_flux)
            else:
                features['transit_dip_duration_ratio'] = 0.0
            
            # Periodicity score (how regular are the dips)
            if len(peaks) > 1:
                intervals = np.diff(peaks)
                if len(intervals) > 1:
                    interval_std = np.std(intervals)
                    interval_mean = np.mean(intervals)
                    if interval_mean > 0:
                        features['transit_periodicity_score'] = 1.0 / (1.0 + interval_std / interval_mean)
                    else:
                        features['transit_periodicity_score'] = 0.0
                else:
                    features['transit_periodicity_score'] = 0.5
            else:
                features['transit_periodicity_score'] = 0.0
        else:
            features['transit_deepest_dip'] = 0.0
            features['transit_dip_duration_ratio'] = 0.0
            features['transit_periodicity_score'] = 0.0
            
    except Exception as e:
        features = {f'transit_{key}': 0.0 for key in [
            'num_dips', 'deepest_dip', 'dip_duration_ratio', 'periodicity_score'
        ]}
    
    return features

def extract_all_features(flux: np.ndarray, time: Optional[np.ndarray] = None, 
                        sample_rate: float = 1.0) -> Dict[str, float]:
    """
    Extract all features from light curve data.
    
    Parameters:
    -----------
    flux : np.ndarray
        Flux values
    time : np.ndarray, optional
        Time values
    sample_rate : float
        Sampling rate for frequency analysis
        
    Returns:
    --------
    dict
        Dictionary containing all extracted features
    """
    all_features = {}
    
    # Extract different types of features
    all_features.update(extract_statistical_features(flux))
    all_features.update(extract_variability_features(flux, time))
    all_features.update(extract_frequency_features(flux, time, sample_rate))
    all_features.update(extract_transit_features(flux, time))
    
    return all_features

def build_lightcurve_features(lightcurve_dir):
    """
    Build features from light curve data stored in CSV files.

    Parameters:
    -----------
    lightcurve_dir : str
        Directory containing light curve CSV files.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the extracted features from all light curves.
    """
    features_list = []

    for filename in os.listdir(lightcurve_dir):
        if filename.endswith('.csv'):
            kepler_id = filename.split('_')[1]  # Extract Kepler ID from filename
            df = pd.read_csv(os.path.join(lightcurve_dir, filename))

            # Extract flux data
            if 'pdcsap_flux' in df.columns:
                flux = df['pdcsap_flux'].values
            elif 'sap_flux' in df.columns:
                flux = df['sap_flux'].values
            elif 'flux' in df.columns:
                flux = df['flux'].values
            else:
                continue
            
            # Remove NaN values
            flux = flux[~np.isnan(flux)]
            
            if len(flux) == 0:
                continue
            
            # Extract comprehensive features
            all_features = extract_all_features(flux)
            all_features['kepler_id'] = kepler_id
            
            features_list.append(all_features)

    # Create a DataFrame from the features list
    features_df = pd.DataFrame(features_list)
    return features_df

def build_koi_features(koi_file):
    """
    Build features from the KOI dataset.

    Parameters:
    -----------
    koi_file : str
        Path to the KOI dataset CSV file.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the features from the KOI dataset.
    """
    koi_df = pd.read_csv(koi_file, comment='#')

    # Select relevant features
    feature_columns = [
        'kepid', 'koi_period', 'koi_duration', 'koi_depth', 'koi_model_snr',
        'koi_impact', 'koi_sma', 'koi_incl', 'koi_steff', 'koi_slogg',
        'koi_srad', 'koi_smass', 'koi_kepmag', 'koi_disposition'
    ]
    
    # Filter to only include columns that exist
    available_features = [col for col in feature_columns if col in koi_df.columns]
    
    selected_features = koi_df[available_features].copy()
    return selected_features

def combine_features(lightcurve_features, koi_features):
    """
    Combine light curve features with KOI features.

    Parameters:
    -----------
    lightcurve_features : pd.DataFrame
        DataFrame containing features from light curves.
    koi_features : pd.DataFrame
        DataFrame containing features from the KOI dataset.

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with both light curve and KOI features.
    """
    # Convert kepler_id to int in lightcurve_features for merging
    lightcurve_features['kepler_id'] = lightcurve_features['kepler_id'].astype(int)
    
    combined_df = pd.merge(lightcurve_features, koi_features, 
                          left_on='kepler_id', right_on='kepid', how='inner')
    return combined_df

def main():
    lightcurve_dir = os.path.join('data', 'raw', 'lightkurve_data')
    koi_file = os.path.join('data', 'raw', 'KOI Selected 2000 Signals.csv')

    try:
        # Build features
        print("Extracting lightcurve features...")
        lightcurve_features = build_lightcurve_features(lightcurve_dir)
        print(f"Extracted features for {len(lightcurve_features)} lightcurves")
        
        print("Loading KOI features...")
        koi_features = build_koi_features(koi_file)
        print(f"Loaded KOI features for {len(koi_features)} objects")

        # Combine features
        print("Combining features...")
        combined_features = combine_features(lightcurve_features, koi_features)
        print(f"Combined dataset has {len(combined_features)} objects")

        # Save combined features to CSV
        output_path = os.path.join('data', 'processed', 'combined_features.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_features.to_csv(output_path, index=False)
        print(f"Combined features saved to '{output_path}'.")
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()