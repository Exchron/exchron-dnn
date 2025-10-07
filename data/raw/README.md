# README for Raw Data

## Overview
This directory contains the raw data used for the exoplanet classification project. The data consists of light curve time series data for various Kepler IDs, which are essential for training the deep learning model to classify exoplanets.

## Data Source
The light curve data is sourced from the NASA Kepler mission, which collected photometric data of stars to detect exoplanets through the transit method. Each CSV file in this directory is named after the corresponding Kepler ID (KIC) and contains time series data representing the brightness of the star over time.

## Data Format
Each CSV file follows the structure outlined below:
- **Time**: The time of observation (in days).
- **Flux**: The measured brightness of the star at the corresponding time.
- **Flux Error**: The uncertainty in the measured brightness.

## Preprocessing Steps
Before using the raw data for model training, the following preprocessing steps may be necessary:
1. **Normalization**: Scale the flux values to a standard range (e.g., 0 to 1) to improve model convergence.
2. **Handling Missing Values**: Identify and appropriately handle any missing or anomalous data points.
3. **Feature Extraction**: Extract relevant features from the light curves, such as period, amplitude, and other statistical measures, which will be used in the model training phase.

## Usage
The raw data files can be loaded using the data loader module in the `src/data/data_loader.py` file. Ensure that the preprocessing steps are applied before training the model to achieve optimal performance.

For further details on feature engineering and model training, refer to the corresponding Jupyter notebooks in the `notebooks` directory.