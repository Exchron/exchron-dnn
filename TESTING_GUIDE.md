# Individual Prediction Testing Scripts

This directory contains two Python scripts for testing individual predictions using the trained `.keras` exoplanet classification model.

## Scripts

### 1. `test_individual_predictions.py`
Basic script that tests the first 5 records from the test set.

**Usage:**
```bash
python test_individual_predictions.py
```

**Features:**
- Tests exactly 5 samples from the beginning of the test set
- Shows detailed prediction analysis for each sample
- Displays key astronomical features
- Provides summary statistics
- Saves results to `logs/individual_predictions_test.csv`

### 2. `test_keras_predictions.py` ⭐ **RECOMMENDED**
Enhanced script with command-line options for flexible testing.

**Usage:**
```bash
# Test first 5 samples (default)
python test_keras_predictions.py

# Test 10 samples starting from index 0
python test_keras_predictions.py -n 10

# Test 5 samples starting from index 20
python test_keras_predictions.py -n 5 --start_idx 20

# Test 15 samples starting from index 50
python test_keras_predictions.py -n 15 --start_idx 50
```

**Command Line Options:**
- `-n, --num_samples`: Number of samples to test (default: 5)
- `--start_idx`: Starting index in test set (default: 0)
- `-h, --help`: Show help message

**Features:**
- ✅ Flexible sample selection with command-line arguments
- 🎯 Enhanced visual output with emojis and clear formatting
- 📊 Detailed astronomical feature analysis
- 📈 Comprehensive summary statistics including confidence scores
- 💾 Automatic results saving with descriptive filenames
- 🔍 Distribution analysis of predictions

## Example Output

```
🚀 EXOPLANET CLASSIFICATION INDIVIDUAL TESTING
Testing 5 samples starting from index 0
Using trained .keras model
============================================================

==================================================
📊 SAMPLE 1
==================================================
🔭 Kepler ID: 8621353
📋 KOI Disposition: FALSE POSITIVE
🏷️  Actual: FALSE POSITIVE

🤖 Model Prediction:
   Probability: 0.008677
   Predicted: FALSE POSITIVE
   Confidence: 0.991323

✅ Result: CORRECT

🌟 Key Features:
   Period: 11.344 days
   Transit Depth: 258240.000 ppm
   Duration: 5.974 hours
   SNR: 437.200
   Impact Parameter: 0.692
   Stellar Temperature: 5780.000 K
```

## Model Information

The scripts automatically load:
- **Model:** `models/dual_input_dnn_model.keras`
- **Feature Scaler:** `models/feature_scaler.pkl` 
- **Feature Names:** `models/feature_names.pkl`

**Model Architecture:**
- Input 1: Time series data (3000 time points)
- Input 2: Engineered features (12 features)
- Output: Binary classification probability (CANDIDATE vs FALSE POSITIVE)

## Features Used

The model uses these 12 engineered features:
1. `koi_period` - Orbital period (days)
2. `koi_duration` - Transit duration (hours)
3. `koi_depth` - Transit depth (ppm)
4. `koi_model_snr` - Signal-to-noise ratio
5. `koi_impact` - Impact parameter
6. `koi_sma` - Semi-major axis (AU)
7. `koi_incl` - Inclination (degrees)
8. `koi_steff` - Stellar effective temperature (K)
9. `koi_slogg` - Stellar surface gravity (log g)
10. `koi_srad` - Stellar radius (solar radii)
11. `koi_smass` - Stellar mass (solar masses)
12. `koi_kepmag` - Kepler magnitude

## Output Files

Results are automatically saved to:
- `logs/individual_predictions_test.csv` (basic script)
- `logs/individual_test_results_{start}_{end}.csv` (enhanced script)

## Requirements

- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Notes

⚠️ **Important:** The scripts require the trained model files to be present in the `models/` directory:
- `dual_input_dnn_model.keras`
- `feature_scaler.pkl`
- `feature_names.pkl`

🔍 **Light Curve Data:** The scripts automatically load light curve data from `data/raw/lightkurve_data/` directory. If a light curve file is missing for a particular Kepler ID, the script will use a zero-filled array and display a warning.

📊 **Test Data:** Uses the preprocessed test data from `data/processed/test_data.csv` which contains 182 samples total.

## Example Use Cases

1. **Quick Test:** `python test_keras_predictions.py`
   - Tests first 5 samples for a quick model check

2. **Batch Testing:** `python test_keras_predictions.py -n 20`
   - Test 20 samples to get better accuracy statistics

3. **Specific Range:** `python test_keras_predictions.py -n 10 --start_idx 100`
   - Test samples 101-110 from the test set

4. **Full Test Set:** `python test_keras_predictions.py -n 182`
   - Test all 182 samples in the test set (takes longer)