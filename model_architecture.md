# Exoplanet Classification Model Architecture - Updated

## Overview
This document provides a comprehensive description of the dual-input deep neural network architecture designed for exoplanet candidate classification.

**Last Updated:** October 7, 2025  
**Modification:** Updated to softmax output for dual probability classification

## Model Summary

- **Model Type:** Dual-Input Deep Neural Network (DNN) with Softmax Output
- **Architecture:** Combined CNN and Dense layers for time series and tabular data
- **Total Parameters:** 3,151,938
- **Trainable Parameters:** 3,150,850
- **Non-trainable Parameters:** 1,088
- **Final Test Accuracy:** 0.9725
- **Test Precision:** 0.9643
- **Test Recall:** 0.9759
- **AUC Score:** 0.9965

## Key Modifications

- ✅ **Output Layer:** Changed from `Dense(1, sigmoid)` to `Dense(2, softmax)`
- ✅ **Loss Function:** Updated from `binary_crossentropy` to `sparse_categorical_crossentropy`
- ✅ **Output Format:** Now provides probabilities for both classes
- ✅ **Performance:** Improved accuracy from 96.15% to 97.25%

## Architecture Details

### Input Specifications

#### 1. Time Series Input (Light Curves)
- **Shape:** `(3000, 1)`
- **Description:** Normalized light curve data representing flux measurements over time
- **Preprocessing:** 
  - Sequence padding/truncation to fixed length
  - Normalization using StandardScaler
  - Missing value handling

#### 2. Stellar and Planetary Features Input
- **Shape:** `(12,)`
- **Features:** 12 KOI (Kepler Objects of Interest) parameters
- **Description:** Stellar and planetary parameters from Kepler mission:
  - **koi_period**: Orbital period (days)
  - **koi_duration**: Transit duration (hours)
  - **koi_depth**: Transit depth (ppm)
  - **koi_model_snr**: Signal-to-noise ratio
  - **koi_impact**: Impact parameter
  - **koi_sma**: Semi-major axis (AU)
  - **koi_incl**: Orbital inclination (degrees)
  - **koi_steff**: Stellar effective temperature (K)
  - **koi_slogg**: Stellar surface gravity (log g)
  - **koi_srad**: Stellar radius (solar radii)
  - **koi_smass**: Stellar mass (solar masses)
  - **koi_kepmag**: Kepler magnitude

### Network Architecture

### Layer Details

#### Layer 1: time_series_input (InputLayer)
- **Type:** InputLayer
- **Output Shape:** Variable
- **Parameters:** 0

#### Layer 2: reshape (Reshape)
- **Type:** Reshape
- **Output Shape:** Variable
- **Parameters:** 0

#### Layer 3: conv1d (Conv1D)
- **Type:** Conv1D
- **Output Shape:** Variable
- **Parameters:** 256
- **Activation:** relu

#### Layer 4: batch_normalization (BatchNormalization)
- **Type:** BatchNormalization
- **Output Shape:** Variable
- **Parameters:** 128

#### Layer 5: max_pooling1d (MaxPooling1D)
- **Type:** MaxPooling1D
- **Output Shape:** Variable
- **Parameters:** 0

#### Layer 6: dropout (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.25

#### Layer 7: conv1d_1 (Conv1D)
- **Type:** Conv1D
- **Output Shape:** Variable
- **Parameters:** 10,304
- **Activation:** relu

#### Layer 8: batch_normalization_1 (BatchNormalization)
- **Type:** BatchNormalization
- **Output Shape:** Variable
- **Parameters:** 256

#### Layer 9: max_pooling1d_1 (MaxPooling1D)
- **Type:** MaxPooling1D
- **Output Shape:** Variable
- **Parameters:** 0

#### Layer 10: dropout_1 (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.3

#### Layer 11: conv1d_2 (Conv1D)
- **Type:** Conv1D
- **Output Shape:** Variable
- **Parameters:** 24,704
- **Activation:** relu

#### Layer 12: batch_normalization_2 (BatchNormalization)
- **Type:** BatchNormalization
- **Output Shape:** Variable
- **Parameters:** 512

#### Layer 13: feature_input (InputLayer)
- **Type:** InputLayer
- **Output Shape:** Variable
- **Parameters:** 0

#### Layer 14: max_pooling1d_2 (MaxPooling1D)
- **Type:** MaxPooling1D
- **Output Shape:** Variable
- **Parameters:** 0

#### Layer 15: dense_2 (Dense)
- **Type:** Dense
- **Output Shape:** Variable
- **Parameters:** 832
- **Units:** 64
- **Activation:** relu

#### Layer 16: dropout_2 (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.4

#### Layer 17: batch_normalization_3 (BatchNormalization)
- **Type:** BatchNormalization
- **Output Shape:** Variable
- **Parameters:** 256

#### Layer 18: flatten (Flatten)
- **Type:** Flatten
- **Output Shape:** Variable
- **Parameters:** 0

#### Layer 19: dropout_4 (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.3

#### Layer 20: dense (Dense)
- **Type:** Dense
- **Output Shape:** Variable
- **Parameters:** 3,047,680
- **Units:** 256
- **Activation:** relu

#### Layer 21: dense_3 (Dense)
- **Type:** Dense
- **Output Shape:** Variable
- **Parameters:** 2,080
- **Units:** 32
- **Activation:** relu

#### Layer 22: dropout_3 (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.5

#### Layer 23: batch_normalization_4 (BatchNormalization)
- **Type:** BatchNormalization
- **Output Shape:** Variable
- **Parameters:** 128

#### Layer 24: dense_1 (Dense)
- **Type:** Dense
- **Output Shape:** Variable
- **Parameters:** 32,896
- **Units:** 128
- **Activation:** relu

#### Layer 25: dropout_5 (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.3

#### Layer 26: concatenate (Concatenate)
- **Type:** Concatenate
- **Output Shape:** Variable
- **Parameters:** 0

#### Layer 27: dense_4 (Dense)
- **Type:** Dense
- **Output Shape:** Variable
- **Parameters:** 20,608
- **Units:** 128
- **Activation:** relu

#### Layer 28: batch_normalization_5 (BatchNormalization)
- **Type:** BatchNormalization
- **Output Shape:** Variable
- **Parameters:** 512

#### Layer 29: dropout_6 (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.4

#### Layer 30: dense_5 (Dense)
- **Type:** Dense
- **Output Shape:** Variable
- **Parameters:** 8,256
- **Units:** 64
- **Activation:** relu

#### Layer 31: batch_normalization_6 (BatchNormalization)
- **Type:** BatchNormalization
- **Output Shape:** Variable
- **Parameters:** 256

#### Layer 32: dropout_7 (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.4

#### Layer 33: dense_6 (Dense)
- **Type:** Dense
- **Output Shape:** Variable
- **Parameters:** 2,080
- **Units:** 32
- **Activation:** relu

#### Layer 34: batch_normalization_7 (BatchNormalization)
- **Type:** BatchNormalization
- **Output Shape:** Variable
- **Parameters:** 128

#### Layer 35: dropout_8 (Dropout)
- **Type:** Dropout
- **Output Shape:** Variable
- **Parameters:** 0
- **Dropout Rate:** 0.4

#### Layer 36: output (Dense)
- **Type:** Dense
- **Output Shape:** (None, 2)
- **Parameters:** 66
- **Units:** 2
- **Activation:** softmax
- **Description:** Outputs probability distribution over two classes:
  - Class 0: Non-candidate (False Positive)
  - Class 1: Candidate (Exoplanet)


### Training Configuration

- **Time Series Length:** 3000
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Max Epochs:** 100
- **Early Stopping Patience:** 15
- **Validation Split:** 0.2
- **Test Split:** 0.15

### Model Compilation

- **Optimizer:** Adam (learning_rate=0.001)
- **Loss Function:** sparse_categorical_crossentropy
- **Metrics:** accuracy
- **Early Stopping:** patience=15, monitor='val_loss'
- **Learning Rate Reduction:** factor=0.5, patience=10

### Performance Metrics

| Metric | Value |
|--------|--------|
| Test Accuracy | 0.9725 |
| Test Precision | 0.9643 |
| Test Recall | 0.9759 |
| AUC Score | 0.9965 |
| Training Samples | 783 |
| Validation Samples | 242 |
| Test Samples | 182 |

## Model Architecture Rationale

### Dual-Input Design
The model employs a dual-input architecture to leverage both temporal patterns in light curves and stellar/planetary features:

1. **CNN Branch:** Processes sequential light curve data using 1D convolutions to capture temporal patterns and transit signatures
2. **Dense Branch:** Processes KOI features to capture stellar and planetary characteristics
3. **Fusion Layer:** Concatenates both representations for combined analysis
4. **Classification Layer:** Softmax output provides explicit probabilities for both classes

### Key Design Decisions

1. **CNN for Time Series:** Multi-scale 1D convolutions capture transit patterns and variability at different temporal scales
2. **KOI Features:** Incorporates validated astronomical parameters from Kepler mission data
3. **Batch Normalization:** Stabilizes training and improves convergence
4. **Dropout Regularization:** Prevents overfitting in the high-dimensional feature space
5. **Softmax Output:** Provides explicit probabilities for both candidate and non-candidate classes
6. **Sparse Categorical Loss:** Efficient training with integer labels while maintaining probability outputs

## Data Pipeline

### Preprocessing Steps
1. Light curve normalization and cleaning
2. Sequence padding/truncation to fixed length
3. Feature engineering from raw time series
4. Train/validation/test splitting with stratification
5. Feature scaling using StandardScaler

### Augmentation Techniques
- Temporal shifts and scaling (if implemented)
- Noise injection for robustness
- Class balancing strategies

## Model Files

- **Keras Format:** `models/dual_input_dnn_model.keras` (Recommended)
- **H5 Format:** `models/dual_input_dnn_model.h5` (Compatibility)
- **Architecture Summary:** `models/model_architecture_summary.txt`
- **Training Logs:** `logs/training_history.png`

## Output Format

The model now outputs a probability distribution over two classes:

```json
{
  "candidate_probability": 0.7389113903045654,
  "non_candidate_probability": 0.26108860969543457
}
```

- **candidate_probability**: Probability that the signal represents a true exoplanet candidate
- **non_candidate_probability**: Probability that the signal is a false positive
- **Sum**: Both probabilities always sum to 1.0 (softmax normalization)

## Usage Instructions

### Basic Usage
```python
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('models/dual_input_dnn_model.keras')

# Make predictions (returns probabilities for both classes)
predictions = model.predict([time_series_data, koi_features])

# Extract probabilities
for i, pred in enumerate(predictions):
    non_candidate_prob = pred[0]
    candidate_prob = pred[1]
    print(f"Sample {i+1}:")
    print(f"  Candidate probability: {candidate_prob:.6f}")
    print(f"  Non-candidate probability: {non_candidate_prob:.6f}")
```

### Using the Prediction API
```python
from prediction_api import ExoplanetPredictor

# Initialize predictor
predictor = ExoplanetPredictor()

# Example KOI features
features = {
    'koi_period': 365.25,
    'koi_duration': 6.5,
    'koi_depth': 84.0,
    'koi_model_snr': 25.0,
    # ... other features
}

# Make prediction
result = predictor.predict_from_kepid(kepid=10000490, feature_data=features)
print(f"Candidate probability: {result['candidate_probability']:.6f}")
print(f"Non-candidate probability: {result['non_candidate_probability']:.6f}")
```

## References

- Kepler Space Telescope Data
- NASA Exoplanet Archive
- TensorFlow/Keras Documentation

---
*This documentation was automatically generated by the Exoplanet Classification Training Pipeline.*
