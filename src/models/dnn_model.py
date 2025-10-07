import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Any, Optional

def create_time_series_branch(input_shape: Tuple[int], dropout_rates: list) -> Model:
    """
    Create the time series processing branch of the neural network.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the time series input (time_steps,)
    dropout_rates : list
        Dropout rates for different layers
        
    Returns:
    --------
    Model
        Time series processing branch
    """
    # Input layer
    time_series_input = Input(shape=input_shape, name='time_series_input')
    
    # Reshape for Conv1D layers
    x = layers.Reshape((input_shape[0], 1))(time_series_input)
    
    # First Conv1D block
    x = layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(dropout_rates[0])(x)
    
    # Second Conv1D block
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(dropout_rates[1])(x)
    
    # Third Conv1D block
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rates[2])(x)
    
    # Flatten for dense layers
    x = layers.Flatten()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rates[3])(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Create model
    time_series_branch = Model(inputs=time_series_input, outputs=x, name='time_series_branch')
    
    return time_series_branch

def create_feature_branch(input_shape: Tuple[int], dense_layers: list, dropout_rate: float = 0.3) -> Model:
    """
    Create the feature processing branch of the neural network.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the feature input (num_features,)
    dense_layers : list
        List of dense layer sizes
    dropout_rate : float
        Dropout rate
        
    Returns:
    --------
    Model
        Feature processing branch
    """
    # Input layer
    feature_input = Input(shape=input_shape, name='feature_input')
    
    x = feature_input
    
    # Dense layers
    for units in dense_layers:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Create model
    feature_branch = Model(inputs=feature_input, outputs=x, name='feature_branch')
    
    return feature_branch

def create_dual_input_dnn_model(time_series_shape: Tuple[int], feature_shape: Tuple[int], 
                               config: Any) -> Model:
    """
    Create a dual-input DNN model combining time series and feature branches.
    
    Parameters:
    -----------
    time_series_shape : tuple
        Shape of time series input
    feature_shape : tuple
        Shape of feature input
    config : Config
        Configuration object with model parameters
        
    Returns:
    --------
    Model
        Compiled dual-input model
    """
    # Create branches
    time_series_branch = create_time_series_branch(time_series_shape, config.dropout_rates)
    feature_branch = create_feature_branch(feature_shape, config.feature_dense_layers)
    
    # Combine branches
    combined = layers.concatenate([time_series_branch.output, feature_branch.output])
    
    # Combined dense layers
    x = combined
    for units in config.combined_dense_layers:
        x = layers.Dense(units, activation=config.activation_function)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
    
    # Output layer - Modified to output probabilities for both classes
    output = layers.Dense(2, activation='softmax', name='output')(x)
    
    # Create final model
    model = Model(
        inputs=[time_series_branch.input, feature_branch.input],
        outputs=output,
        name='dual_input_exoplanet_classifier'
    )
    
    # Compile model
    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_callbacks(config: Any, monitor: str = 'val_loss') -> list:
    """
    Create training callbacks.
    
    Parameters:
    -----------
    config : Config
        Configuration object
    monitor : str
        Metric to monitor
        
    Returns:
    --------
    list
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model checkpoint
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.checkpoint_path, 'best_model.h5'),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    return callbacks

def train_model(model: Model, train_data: Tuple, val_data: Tuple, config: Any) -> Dict[str, Any]:
    """
    Train the dual-input model.
    
    Parameters:
    -----------
    model : Model
        Compiled model
    train_data : tuple
        (train_sequences, train_features, train_labels)
    val_data : tuple
        (val_sequences, val_features, val_labels)
    config : Config
        Configuration object
        
    Returns:
    --------
    dict
        Training history and metrics
    """
    train_sequences, train_features, train_labels = train_data
    val_sequences, val_features, val_labels = val_data
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    print("Starting model training...")
    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    
    # Train model
    history = model.fit(
        x=[train_sequences, train_features],
        y=train_labels,
        validation_data=([val_sequences, val_features], val_labels),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=config.verbose
    )
    
    # Save final model
    if not os.path.exists(os.path.dirname(config.model_save_path)):
        os.makedirs(os.path.dirname(config.model_save_path))
    
    model.save(config.model_save_path)
    print(f"Model saved to {config.model_save_path}")
    
    return history

def evaluate_model(model: Model, test_data: Tuple, feature_names: list) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Parameters:
    -----------
    model : Model
        Trained model
    test_data : tuple
        (test_sequences, test_features, test_labels)
    feature_names : list
        Names of features
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    test_sequences, test_features, test_labels = test_data
    
    # Make predictions
    predictions = model.predict([test_sequences, test_features])
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_probs = predictions[:, 1]  # Probability of candidate class
    predicted_probs_both = predictions  # Both probabilities for detailed analysis
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(
        [test_sequences, test_features], test_labels, verbose=0
    )
    
    # Calculate precision and recall manually for the positive class
    from sklearn.metrics import precision_score, recall_score
    test_precision = precision_score(test_labels, predicted_classes, zero_division=0)
    test_recall = recall_score(test_labels, predicted_classes, zero_division=0)
    
    auc_score = roc_auc_score(test_labels, predicted_probs)
    
    # Classification report
    class_report = classification_report(test_labels, predicted_classes, 
                                       target_names=['False Positive', 'Candidate'])
    
    # Confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_classes)
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'auc_score': auc_score,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'predicted_probs': predicted_probs,
        'predicted_probs_both': predicted_probs_both
    }
    
    print(f"\\nModel Evaluation Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"\\nClassification Report:")
    print(class_report)
    print(f"\\nConfusion Matrix:")
    print(conf_matrix)
    
    return results

def plot_training_history(history: Any, save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Parameters:
    -----------
    history : History
        Training history from model.fit()
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot training & validation precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot training & validation recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_roc_curve(test_labels: np.ndarray, predicted_probs: np.ndarray, 
                  save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    test_labels : np.ndarray
        True labels
    predicted_probs : np.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save the plot
    """
    fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
    auc_score = roc_auc_score(test_labels, predicted_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function for testing the model creation.
    """
    from ..config import config
    
    # Example shapes
    time_series_shape = (config.time_series_length,)
    feature_shape = (config.num_features,)
    
    # Create model
    model = create_dual_input_dnn_model(time_series_shape, feature_shape, config)
    
    # Print model summary
    print("Model created successfully!")
    print(f"Time series input shape: {time_series_shape}")
    print(f"Feature input shape: {feature_shape}")
    
    model.summary()

if __name__ == "__main__":
    main()