# IMPORTANT
# Due to time limitations, the README.md file provides an AI generated summary of the project. For detailed instructions and methodology please check back later or refer to the https://docs.exchronai.earth

# Exoplanet Classification with Dual-Input Deep Neural Network

This project implements a sophisticated exoplanet classification system using a dual-input Deep Neural Network (DNN) architecture. The model combines light curve time series data with extracted features to accurately predict whether a Kepler Object of Interest (KOI) is a confirmed exoplanet candidate or a false positive.

## 🌟 Key Features

- **Dual-Input Architecture**: Combines CNN processing of time series data with traditional feature processing
- **Advanced Feature Engineering**: Extracts statistical, variability, frequency, and transit features from light curves
- **SHAP Explainability**: Provides model interpretability and feature importance analysis
- **Robust Data Pipeline**: Handles missing data, outliers, and data quality issues automatically
- **Complete Training Pipeline**: End-to-end solution from raw data to trained model

## 🏗️ Model Architecture

The DNN architecture features two specialized branches:

### Time Series Branch
- 1D Convolutional layers for pattern detection in light curves
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Processes normalized flux sequences (3000 time points)

### Feature Branch
- Dense layers for traditional feature processing
- Handles extracted statistical, variability, and astronomical features
- Batch normalization and dropout for stability

### Combined Architecture
- Both branches merge for final classification
- Dense layers with regularization
- Binary output (Candidate vs False Positive)

## 📁 Project Structure

```
exoplanet-classification/
├── train.py                     # Main training script
├── configs/
│   └── model_config.yaml        # Model configuration
├── data/
│   ├── raw/
│   │   ├── KOI Selected 2000 Signals.csv
│   │   └── lightkurve_data/     # Individual light curve files
│   └── processed/               # Generated during training
├── src/
│   ├── config.py               # Configuration management
│   ├── data/
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── preprocessing.py    # Complete preprocessing pipeline
│   ├── features/
│   │   ├── build_features.py   # Feature extraction
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── dnn_model.py        # Dual-input DNN architecture
│   │   └── train_model.py      # Training orchestration
│   └── visualization/
│       ├── explainability.py   # SHAP integration
│       └── visualize.py        # Plotting utilities
└── notebooks/                  # Jupyter notebooks for exploration
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages include:
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- SHAP
- matplotlib
- seaborn
- scipy

### Training the Model

Run the complete training pipeline:

```bash
python train.py
```

This will:
1. Load and preprocess light curve data
2. Extract features from time series
3. Create train/validation/test splits
4. Build and train the dual-input model
5. Evaluate performance and generate SHAP explanations

### Configuration

Modify `configs/model_config.yaml` to adjust:
- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Feature extraction options

## 📊 Data Processing

### Light Curve Processing
- **Input**: CSV files with flux measurements over time
- **Normalization**: Robust scaling using median and MAD
- **Outlier Removal**: Sigma clipping for noise reduction
- **Sequence Length**: Fixed to 3000 points (padded/truncated)

### Feature Extraction
The system extracts 24+ features including:
- **Statistical**: Mean, std, median, skewness, kurtosis
- **Variability**: Amplitude, flux excursions, range
- **Frequency**: Dominant frequency, spectral features
- **Transit**: Dip detection, periodicity analysis

### KOI Integration
- Astronomical parameters from KOI catalog
- Missing value imputation
- Feature scaling and normalization

## 🔍 Model Interpretability

The project includes comprehensive explainability features:

- **SHAP Values**: Feature importance for individual predictions
- **Feature Ranking**: Global feature importance across dataset
- **Visualization**: Interactive plots and explanations
- **Comparison Plots**: Time series vs feature importance

## 📈 Performance Metrics

The model is evaluated using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and Precision-Recall curves
- Confusion matrices
- Cross-validation scores

## 🔧 Advanced Usage

### Custom Feature Engineering
```python
from src.features.build_features import extract_all_features
features = extract_all_features(lightcurve_data)
```

### Model Customization
```python
from src.models.dnn_model import create_dual_input_dnn_model
model = create_dual_input_dnn_model(
    sequence_length=3000,
    feature_dim=36,
    architecture_config=config
)
```

### Batch Prediction
```python
from src.data.preprocessing import load_and_preprocess_data
sequences, features, labels = load_and_preprocess_data()
predictions = model.predict([sequences, features])
```

## 📚 Notebooks

Explore the project through interactive notebooks:
- `01_data_exploration.ipynb`: Data analysis and visualization
- `02_feature_engineering.ipynb`: Feature extraction deep dive
- `03_model_training.ipynb`: Model development and training
- `04_model_evaluation.ipynb`: Performance analysis and interpretation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- NASA Kepler mission for providing the light curve data
- KOI catalog for astronomical parameters
- TensorFlow and scikit-learn communities

## 📞 Support

For questions or issues:
1. Check the documentation in each module
2. Review the example notebooks
3. Open an issue on the repository

---

**Ready to discover exoplanets? Run `python train.py` to get started!** 🌍✨