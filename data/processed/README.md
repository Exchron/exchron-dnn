# Processed Data Directory

This directory contains the processed and transformed data used for training the exoplanet classification model.

## Data Processing Pipeline

The data processing pipeline performs the following steps:

### 1. Light Curve Processing
- **Input**: Raw light curve CSV files from `data/raw/lightkurve_data/`
- **Processing Steps**:
  - Extract flux values (prioritizing `pdcsap_flux`, then `sap_flux`, then `flux`)
  - Remove NaN values and outliers using sigma clipping
  - Normalize flux using robust scaling (median and median absolute deviation)
  - Pad or truncate sequences to a fixed length (default: 3000 points)
  - Apply additional filtering and smoothing if needed

### 2. Feature Extraction
- **Statistical Features**: Mean, std, median, skewness, kurtosis, percentiles
- **Variability Features**: Amplitude, flux excursions, variability metrics
- **Frequency Features**: Dominant frequency, spectral centroid, spectral rolloff
- **Transit Features**: Number of dips, deepest dip, periodicity score

### 3. KOI Feature Integration
- **Input**: `data/raw/KOI Selected 2000 Signals.csv`
- **Processing**:
  - Extract relevant astronomical features (period, duration, depth, etc.)
  - Handle missing values using median imputation
  - Normalize features using StandardScaler
  - Create binary labels from disposition (CANDIDATE vs FALSE POSITIVE)

### 4. Train/Validation/Test Split
- **Strategy**: Stratified split maintaining class balance
- **Default Split**: 65% train, 20% validation, 15% test
- **Output**: Separate arrays for sequences, features, and labels

## Generated Files

When the preprocessing pipeline runs, it generates the following files:

- `combined_features.csv`: Combined light curve and KOI features
- `feature_scaler.pkl`: Fitted StandardScaler for feature normalization
- `feature_names.pkl`: List of feature names for interpretation
- `train_sequences.npy`: Training light curve sequences
- `val_sequences.npy`: Validation light curve sequences  
- `test_sequences.npy`: Test light curve sequences
- `train_features.npy`: Training extracted features
- `val_features.npy`: Validation extracted features
- `test_features.npy`: Test extracted features
- `train_labels.npy`: Training labels
- `val_labels.npy`: Validation labels
- `test_labels.npy`: Test labels

## Data Statistics

After processing, the typical dataset contains:

- **Time Series Data**: 3000-point normalized flux sequences
- **Extracted Features**: ~24 engineered features per light curve
- **KOI Features**: ~12 astronomical parameters
- **Total Samples**: ~1000-1200 objects (depending on data availability)
- **Class Distribution**: Approximately 50% candidates, 50% false positives

## Usage

The processed data is automatically generated when running the training pipeline:

```bash
python train.py
```

To manually run the preprocessing pipeline:

```bash
python src/data/preprocessing.py
```

To extract features from light curves:

```bash
python src/features/build_features.py
```

## Data Quality Checks

The preprocessing pipeline includes several quality checks:

1. **Missing Data**: Light curves with insufficient data points are excluded
2. **Outlier Detection**: Statistical outliers are removed using sigma clipping
3. **Feature Validation**: Features with too many missing values are excluded
4. **Class Balance**: Ensures reasonable balance between candidates and false positives

## Notes

- The processed data is deterministic given the same random seed
- All normalization parameters are saved for consistent inference
- The pipeline is designed to handle missing or corrupted light curve files gracefully
- Feature names are preserved for model interpretability