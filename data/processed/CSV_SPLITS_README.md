# Train/Test Split CSV Files Documentation

## Overview

The data preprocessing pipeline now automatically creates CSV files containing the train/validation/test split information **without the lightcurve data**. This allows you to easily examine and use the metadata and features for each split.

## Generated Files

The following CSV files are created in the `data/processed/` directory:

### 1. `train_data.csv`
- **Purpose:** Contains all training samples with their metadata and features
- **Columns:** 28 total columns including:
  - `kepid`: Kepler ID (identifier)
  - `koi_disposition`: Original KOI disposition (CANDIDATE/FALSE POSITIVE)
  - All KOI features: `koi_period`, `koi_depth`, `koi_model_snr`, etc.
  - `split`: Always "train" for this file
  - `split_label`: Binary label (0=False Positive, 1=Candidate)
  - `label`: Same as split_label (for compatibility)

### 2. `validation_data.csv`
- **Purpose:** Contains all validation samples with their metadata and features
- **Structure:** Same as train_data.csv but with `split`: "validation"

### 3. `test_data.csv`
- **Purpose:** Contains all test samples with their metadata and features
- **Structure:** Same as train_data.csv but with `split`: "test"

### 4. `data_splits_combined.csv`
- **Purpose:** Combined file containing all splits in one file
- **Advantage:** Easy to load and filter by split type
- **Contains:** All samples from train/validation/test with split column indicating which set each sample belongs to

## Sample Data Structure

```
kepid    | koi_disposition | koi_period | koi_depth | split | split_label
---------|-----------------|------------|-----------|-------|------------
4902030  | FALSE POSITIVE  | 0.878803   | 410720.0  | train | 0
3442054  | FALSE POSITIVE  | 117.681378 | 45362.0   | train | 0
5374838  | CANDIDATE       | 230.180357 | 3689.3    | train | 1
```

## Key Features

### ✅ **What's Included:**
- All original KOI metadata and features
- Kepler IDs for cross-referencing
- Split assignment (train/validation/test)
- Binary labels for classification
- All numerical features used in the model

### ❌ **What's NOT Included:**
- **Light curve time series data** (too large for CSV format)
- Processed/normalized feature values (only original values)
- Model predictions or probabilities

## Usage Examples

### Loading Training Data
```python
import pandas as pd

# Load training data
train_df = pd.read_csv('data/processed/train_data.csv')
print(f"Training samples: {len(train_df)}")
print(f"Positive samples: {(train_df['split_label'] == 1).sum()}")
```

### Loading All Splits
```python
# Load combined data and filter by split
all_data = pd.read_csv('data/processed/data_splits_combined.csv')

train_data = all_data[all_data['split'] == 'train']
val_data = all_data[all_data['split'] == 'validation'] 
test_data = all_data[all_data['split'] == 'test']
```

### Cross-referencing with Light Curves
```python
# Get Kepler IDs for a specific split
test_ids = test_data['kepid'].tolist()

# Load corresponding light curves
for kepid in test_ids:
    lightcurve_file = f'data/raw/lightkurve_data/kepler_{kepid}_lightkurve.csv'
    # Process lightcurve...
```

## Statistics

Based on the current dataset split:

- **Training Set:** ~602 samples (45.7% candidates)
- **Validation Set:** ~242 samples (45.5% candidates)  
- **Test Set:** ~363 samples (45.5% candidates)
- **Total:** ~1,207 samples with balanced class distribution

## File Locations

```
data/processed/
├── train_data.csv              # Training split
├── validation_data.csv         # Validation split  
├── test_data.csv              # Test split
└── data_splits_combined.csv   # All splits combined
```

## Integration with Training Pipeline

These CSV files are automatically generated when running:
- `train.py` (full training pipeline)
- `src.data.preprocessing.load_and_preprocess_data()` function

The CSV creation happens **before** lightcurve processing, so it's fast and doesn't require loading all time series data.

## Benefits

1. **Quick Analysis:** Examine data splits without loading large lightcurve files
2. **Reproducibility:** Exact same train/test splits for different experiments
3. **External Tools:** Use with other analysis tools that prefer CSV format
4. **Debugging:** Easy to inspect which samples are in which split
5. **Feature Engineering:** Access to original features for additional processing

---

*Generated automatically by the exoplanet classification preprocessing pipeline.*