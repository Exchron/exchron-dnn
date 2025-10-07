import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(features_file, lightcurve_folder):
    """
    Load features and light curve data for exoplanet classification.

    Parameters:
    -----------
    features_file : str
        Path to the CSV file containing extracted features.
    lightcurve_folder : str
        Path to the folder containing light curve CSV files.

    Returns:
    --------
    pd.DataFrame, pd.DataFrame
        DataFrames containing the features and light curve data.
    """
    features_df = pd.read_csv(features_file)
    
    # Load light curve data
    lightcurve_data = []
    for file in os.listdir(lightcurve_folder):
        if file.endswith('.csv'):
            lc_df = pd.read_csv(os.path.join(lightcurve_folder, file))
            lightcurve_data.append(lc_df)
    
    lightcurve_df = pd.concat(lightcurve_data, ignore_index=True)
    
    return features_df, lightcurve_df

def select_features(X, y, k=10):
    """
    Select the top k features based on ANOVA F-value.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature DataFrame.
    y : pd.Series
        Target variable.
    k : int
        Number of top features to select.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    
    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)
    
    return X.iloc[:, selected_indices]

def preprocess_data(features_df, lightcurve_df):
    """
    Preprocess the data by scaling features and splitting into training and validation sets.

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame containing the features.
    lightcurve_df : pd.DataFrame
        DataFrame containing the light curve data.

    Returns:
    --------
    tuple
        Tuple containing training and validation sets for features and target variable.
    """
    # Assuming the target variable is in the features DataFrame
    X = features_df.drop(columns=['target'])  # Replace 'target' with the actual target column name
    y = features_df['target']  # Replace 'target' with the actual target column name

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

def main():
    features_file = "random-2000/KOI Selected 2000 Signals.csv"
    lightcurve_folder = "lightkurve_data"

    # Load data
    features_df, lightcurve_df = load_data(features_file, lightcurve_folder)

    # Select relevant features
    X_selected = select_features(features_df.drop(columns=['target']), features_df['target'], k=10)

    # Preprocess data
    X_train, X_val, y_train, y_val = preprocess_data(features_df, lightcurve_df)

    # Save selected features and preprocessed data for further use
    X_selected.to_csv("selected_features.csv", index=False)

if __name__ == "__main__":
    main()