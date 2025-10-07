import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Any, Optional
import keras_tuner as kt

def create_attention_layer(input_tensor, units=128):
    """
    Create an attention mechanism for time series data.
    
    Parameters:
    -----------
    input_tensor : tensor
        Input tensor from LSTM layer
    units : int
        Number of attention units
        
    Returns:
    --------
    tensor
        Attention-weighted output
    """
    # Attention weights
    attention = layers.Dense(1, activation='tanh')(input_tensor)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(units)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention to input
    weighted = layers.Multiply()([input_tensor, attention])
    output = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)
    
    return output

def create_enhanced_time_series_branch(input_shape: Tuple[int], hp: kt.HyperParameters = None) -> Model:
    """
    Create an enhanced time series processing branch with multiple architectures.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the time series input (time_steps,)
    hp : kt.HyperParameters
        Hyperparameter object for tuning
        
    Returns:
    --------
    Model
        Enhanced time series processing branch
    """
    if hp is None:
        # Default parameters for non-tuning mode
        class DefaultHP:
            def Choice(self, name, values, default=None):
                return default if default else values[0]
            def Int(self, name, min_value, max_value, default=None):
                return default if default else min_value
            def Float(self, name, min_value, max_value, default=None):
                return default if default else min_value
        hp = DefaultHP()
    
    # Input layer
    time_series_input = Input(shape=input_shape, name='time_series_input')
    x = layers.Reshape((input_shape[0], 1))(time_series_input)
    
    # Architecture choice
    architecture = hp.Choice('ts_architecture', ['cnn_lstm', 'multi_cnn', 'pure_lstm'], default='cnn_lstm')
    
    if architecture == 'cnn_lstm':
        # CNN + LSTM architecture
        # Multi-scale CNN feature extraction
        conv_branches = []
        kernel_sizes = [3, 5, 7, 11]
        
        for kernel_size in kernel_sizes:
            branch = layers.Conv1D(
                filters=hp.Int(f'conv_filters_{kernel_size}', 32, 128, default=64),
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            )(x)
            branch = layers.BatchNormalization()(branch)
            branch = layers.MaxPooling1D(pool_size=2)(branch)
            conv_branches.append(branch)
        
        # Concatenate multi-scale features
        if len(conv_branches) > 1:
            x = layers.concatenate(conv_branches)
        else:
            x = conv_branches[0]
        
        # Additional CNN layers
        num_cnn_layers = hp.Int('num_cnn_layers', 1, 3, default=2)
        for i in range(num_cnn_layers):
            x = layers.Conv1D(
                filters=hp.Int(f'conv_filters_layer_{i}', 64, 256, default=128),
                kernel_size=hp.Choice(f'conv_kernel_{i}', [3, 5, 7], default=3),
                activation='relu',
                padding='same'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=hp.Int(f'pool_size_{i}', 2, 4, default=2))(x)
            x = layers.Dropout(hp.Float(f'conv_dropout_{i}', 0.1, 0.5, default=0.25))(x)
        
        # LSTM layers with attention
        lstm_units = hp.Int('lstm_units', 64, 256, default=128)
        x = layers.LSTM(lstm_units, return_sequences=True)(x)
        x = layers.Dropout(hp.Float('lstm_dropout', 0.1, 0.5, default=0.3))(x)
        
        # Attention mechanism
        x = create_attention_layer(x, lstm_units)
        
    elif architecture == 'multi_cnn':
        # Pure CNN with multiple scales
        # Parallel CNN branches with different kernel sizes
        branches = []
        kernel_sizes = [3, 5, 7, 11, 15]
        
        for kernel_size in kernel_sizes:
            branch = x
            # Multiple conv layers per branch
            for layer_idx in range(hp.Int('cnn_depth', 3, 6, default=4)):
                filters = hp.Int(f'branch_{kernel_size}_filters_{layer_idx}', 32, 128, default=64)
                branch = layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same'
                )(branch)
                branch = layers.BatchNormalization()(branch)
                if layer_idx % 2 == 1:  # Pool every other layer
                    branch = layers.MaxPooling1D(pool_size=2)(branch)
                branch = layers.Dropout(hp.Float(f'branch_dropout_{kernel_size}_{layer_idx}', 0.1, 0.4, default=0.2))(branch)
            
            # Global pooling for each branch
            branch = layers.GlobalAveragePooling1D()(branch)
            branches.append(branch)
        
        # Concatenate all branches
        x = layers.concatenate(branches)
        
    else:  # pure_lstm
        # Pure LSTM architecture with residual connections
        lstm_layers = hp.Int('num_lstm_layers', 2, 4, default=3)
        lstm_units = hp.Int('lstm_units', 64, 256, default=128)
        
        for i in range(lstm_layers):
            if i == lstm_layers - 1:  # Last layer
                lstm_out = layers.LSTM(lstm_units, return_sequences=True)(x)
            else:
                lstm_out = layers.LSTM(lstm_units, return_sequences=True)(x)
                x = layers.Add()([x, lstm_out]) if x.shape[-1] == lstm_out.shape[-1] else lstm_out
            
            x = layers.Dropout(hp.Float(f'lstm_dropout_{i}', 0.1, 0.5, default=0.3))(lstm_out)
        
        # Attention mechanism
        x = create_attention_layer(x, lstm_units)
    
    # Final dense layers
    num_dense = hp.Int('ts_dense_layers', 1, 3, default=2)
    for i in range(num_dense):
        x = layers.Dense(
            hp.Int(f'ts_dense_units_{i}', 64, 512, default=256),
            activation=hp.Choice('ts_activation', ['relu', 'elu', 'swish'], default='relu')
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float(f'ts_dense_dropout_{i}', 0.2, 0.6, default=0.4))(x)
    
    # Create model
    time_series_branch = Model(inputs=time_series_input, outputs=x, name='enhanced_time_series_branch')
    
    return time_series_branch

def create_enhanced_feature_branch(input_shape: Tuple[int], hp: kt.HyperParameters = None) -> Model:
    """
    Create an enhanced feature processing branch.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the feature input (num_features,)
    hp : kt.HyperParameters
        Hyperparameter object for tuning
        
    Returns:
    --------
    Model
        Enhanced feature processing branch
    """
    if hp is None:
        # Default parameters for non-tuning mode
        class DefaultHP:
            def Choice(self, name, values, default=None):
                return default if default else values[0]
            def Int(self, name, min_value, max_value, default=None):
                return default if default else min_value
            def Float(self, name, min_value, max_value, default=None):
                return default if default else min_value
        hp = DefaultHP()
    
    # Input layer
    feature_input = Input(shape=input_shape, name='feature_input')
    
    # Feature engineering layers
    x = feature_input
    
    # Optional feature interaction layer
    if hp.Choice('use_feature_interactions', [True, False], default=False):
        # Create polynomial features (degree 2)
        x_squared = layers.Lambda(lambda x: tf.square(x))(x)
        x = layers.concatenate([x, x_squared])
    
    # Main dense layers with skip connections
    num_layers = hp.Int('feature_layers', 2, 6, default=4)
    layer_outputs = [x]
    
    for i in range(num_layers):
        units = hp.Int(f'feature_units_{i}', 32, 256, default=128)
        
        # Dense layer
        dense_out = layers.Dense(
            units,
            activation=hp.Choice('feature_activation', ['relu', 'elu', 'swish'], default='relu')
        )(x)
        
        # Batch normalization
        dense_out = layers.BatchNormalization()(dense_out)
        
        # Dropout
        dense_out = layers.Dropout(hp.Float(f'feature_dropout_{i}', 0.1, 0.5, default=0.3))(dense_out)
        
        # Skip connection if dimensions match
        if hp.Choice('use_skip_connections', [True, False], default=True) and x.shape[-1] == units:
            x = layers.Add()([x, dense_out])
        else:
            x = dense_out
        
        layer_outputs.append(x)
    
    # Optional attention over layer outputs
    if hp.Choice('use_layer_attention', [True, False], default=False):
        # Simple attention over different layer outputs
        attention_weights = layers.Dense(len(layer_outputs), activation='softmax')(x)
        weighted_outputs = []
        for i, layer_out in enumerate(layer_outputs):
            weight = layers.Lambda(lambda x: x[:, i:i+1])(attention_weights)
            weighted = layers.Multiply()([layer_out, weight])
            weighted_outputs.append(weighted)
        x = layers.Add()(weighted_outputs)
    
    # Create model
    feature_branch = Model(inputs=feature_input, outputs=x, name='enhanced_feature_branch')
    
    return feature_branch

def create_enhanced_dual_input_model(time_series_shape: Tuple[int], feature_shape: Tuple[int], 
                                   hp: kt.HyperParameters = None) -> Model:
    """
    Create an enhanced dual-input DNN model with hyperparameter tuning.
    
    Parameters:
    -----------
    time_series_shape : tuple
        Shape of time series input
    feature_shape : tuple
        Shape of feature input
    hp : kt.HyperParameters
        Hyperparameter object for tuning
        
    Returns:
    --------
    Model
        Enhanced dual-input model
    """
    if hp is None:
        # Default parameters for non-tuning mode
        class DefaultHP:
            def Choice(self, name, values, default=None):
                return default if default else values[0]
            def Int(self, name, min_value, max_value, default=None):
                return default if default else min_value
            def Float(self, name, min_value, max_value, default=None):
                return default if default else min_value
        hp = DefaultHP()
    
    # Create enhanced branches
    time_series_branch = create_enhanced_time_series_branch(time_series_shape, hp)
    feature_branch = create_enhanced_feature_branch(feature_shape, hp)
    
    # Fusion strategy
    fusion_strategy = hp.Choice('fusion_strategy', ['concatenate', 'attention_fusion', 'bilinear'], default='concatenate')
    
    if fusion_strategy == 'concatenate':
        # Simple concatenation
        combined = layers.concatenate([time_series_branch.output, feature_branch.output])
    
    elif fusion_strategy == 'attention_fusion':
        # Cross-attention between branches
        ts_output = time_series_branch.output
        feature_output = feature_branch.output
        
        # Cross attention: time series attending to features
        ts_attention = layers.Dense(feature_output.shape[-1], activation='tanh')(ts_output)
        ts_attention = layers.Dense(feature_output.shape[-1], activation='softmax')(ts_attention)
        ts_attended = layers.Multiply()([feature_output, ts_attention])
        
        # Cross attention: features attending to time series
        feature_attention = layers.Dense(ts_output.shape[-1], activation='tanh')(feature_output)
        feature_attention = layers.Dense(ts_output.shape[-1], activation='softmax')(feature_attention)
        feature_attended = layers.Multiply()([ts_output, feature_attention])
        
        # Combine
        combined = layers.concatenate([ts_attended, feature_attended, ts_output, feature_output])
    
    else:  # bilinear
        # Bilinear fusion
        ts_output = time_series_branch.output
        feature_output = feature_branch.output
        
        # Bilinear layer
        bilinear_dim = hp.Int('bilinear_dim', 64, 256, default=128)
        ts_projected = layers.Dense(bilinear_dim)(ts_output)
        feature_projected = layers.Dense(bilinear_dim)(feature_output)
        
        bilinear = layers.Multiply()([ts_projected, feature_projected])
        combined = layers.concatenate([bilinear, ts_output, feature_output])
    
    # Combined processing layers
    x = combined
    num_combined_layers = hp.Int('combined_layers', 2, 5, default=3)
    
    for i in range(num_combined_layers):
        units = hp.Int(f'combined_units_{i}', 64, 512, default=256)
        x = layers.Dense(
            units,
            activation=hp.Choice('combined_activation', ['relu', 'elu', 'swish'], default='relu')
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float(f'combined_dropout_{i}', 0.2, 0.6, default=0.4))(x)
    
    # Output layer with optional ensemble - Modified for two-class output
    if hp.Choice('use_ensemble_output', [True, False], default=False):
        # Multiple output heads with ensemble
        num_heads = hp.Int('ensemble_heads', 2, 5, default=3)
        head_outputs = []
        
        for i in range(num_heads):
            head = layers.Dense(32, activation='relu')(x)
            head = layers.Dropout(0.3)(head)
            head = layers.Dense(2, activation='softmax', name=f'head_{i}')(head)
            head_outputs.append(head)
        
        # Average ensemble
        output = layers.Average()(head_outputs)
    else:
        # Single output with softmax for two classes
        output = layers.Dense(2, activation='softmax', name='output')(x)
    
    # Create final model
    model = Model(
        inputs=[time_series_branch.input, feature_branch.input],
        outputs=output,
        name='enhanced_dual_input_exoplanet_classifier'
    )
    
    # Compile model with tuned hyperparameters
    optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
    learning_rate = hp.Float('learning_rate', 1e-5, 1e-2, sampling='LOG', default=1e-3)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'AUC']
    )
    
    return model

class ExoplanetHyperModel(kt.HyperModel):
    """
    Keras Tuner HyperModel for exoplanet classification.
    """
    
    def __init__(self, time_series_shape, feature_shape):
        self.time_series_shape = time_series_shape
        self.feature_shape = feature_shape
    
    def build(self, hp):
        """Build model with hyperparameters."""
        return create_enhanced_dual_input_model(self.time_series_shape, self.feature_shape, hp)
    
    def fit(self, hp, model, *args, **kwargs):
        """Custom fit method with hyperparameter-dependent callbacks."""
        # Batch size tuning
        batch_size = hp.Int('batch_size', 16, 128, step=16, default=32)
        kwargs['batch_size'] = batch_size
        
        # Callbacks
        callbacks = kwargs.get('callbacks', [])
        
        # Early stopping with tuned patience
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=hp.Int('early_stopping_patience', 10, 30, default=15),
            restore_best_weights=True,
            mode='max'
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            factor=hp.Float('lr_reduce_factor', 0.1, 0.7, default=0.5),
            patience=hp.Int('lr_reduce_patience', 5, 15, default=10),
            min_lr=1e-7,
            mode='max'
        )
        callbacks.append(reduce_lr)
        
        kwargs['callbacks'] = callbacks
        
        return model.fit(*args, **kwargs)

def run_hyperparameter_tuning(train_data, val_data, time_series_shape, feature_shape, 
                             max_trials=50, executions_per_trial=1, 
                             tuner_type='random', project_name='exoplanet_tuning'):
    """
    Run hyperparameter tuning using Keras Tuner.
    
    Parameters:
    -----------
    train_data : tuple
        (train_sequences, train_features, train_labels)
    val_data : tuple
        (val_sequences, val_features, val_labels)
    time_series_shape : tuple
        Shape of time series input
    feature_shape : tuple
        Shape of feature input
    max_trials : int
        Maximum number of trials
    executions_per_trial : int
        Number of executions per trial
    tuner_type : str
        Type of tuner ('random', 'bayesian', 'hyperband')
    project_name : str
        Project name for tuning directory
        
    Returns:
    --------
    kt.Tuner
        Fitted tuner object
    """
    train_sequences, train_features, train_labels = train_data
    val_sequences, val_features, val_labels = val_data
    
    # Create hypermodel
    hypermodel = ExoplanetHyperModel(time_series_shape, feature_shape)
    
    # Create tuner
    tuner_dir = os.path.join('models', 'hyperparameter_tuning')
    os.makedirs(tuner_dir, exist_ok=True)
    
    if tuner_type == 'random':
        tuner = kt.RandomSearch(
            hypermodel,
            objective=kt.Objective('val_auc', direction='max'),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=tuner_dir,
            project_name=project_name,
            overwrite=True
        )
    elif tuner_type == 'bayesian':
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective=kt.Objective('val_auc', direction='max'),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=tuner_dir,
            project_name=project_name,
            overwrite=True
        )
    else:  # hyperband
        tuner = kt.Hyperband(
            hypermodel,
            objective=kt.Objective('val_auc', direction='max'),
            max_epochs=50,
            factor=3,
            directory=tuner_dir,
            project_name=project_name,
            overwrite=True
        )
    
    print(f"Starting hyperparameter tuning with {tuner_type} search...")
    print(f"Max trials: {max_trials}")
    print(f"Executions per trial: {executions_per_trial}")
    
    # Run search
    tuner.search(
        x=[train_sequences, train_features],
        y=train_labels,
        validation_data=([val_sequences, val_features], val_labels),
        epochs=100,
        verbose=1
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nBest hyperparameters:")
    for param_name, param_value in best_hps.values.items():
        print(f"  {param_name}: {param_value}")
    
    return tuner

def create_best_model_from_tuning(tuner, time_series_shape, feature_shape):
    """
    Create and return the best model from hyperparameter tuning.
    
    Parameters:
    -----------
    tuner : kt.Tuner
        Fitted tuner object
    time_series_shape : tuple
        Shape of time series input
    feature_shape : tuple
        Shape of feature input
        
    Returns:
    --------
    Model
        Best model with optimal hyperparameters
    """
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build best model
    best_model = create_enhanced_dual_input_model(time_series_shape, feature_shape, best_hps)
    
    print("Best model created with optimal hyperparameters.")
    print(f"Model has {best_model.count_params():,} parameters")
    
    return best_model, best_hps

def save_tuning_results(tuner, save_dir='models/tuning_results'):
    """
    Save hyperparameter tuning results.
    
    Parameters:
    -----------
    tuner : kt.Tuner
        Fitted tuner object
    save_dir : str
        Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all trials
    trials = tuner.oracle.get_best_trials(num_trials=tuner.oracle.max_trials)
    
    # Create results DataFrame
    results_data = []
    for trial in trials:
        trial_data = {
            'trial_id': trial.trial_id,
            'score': trial.score,
            'status': trial.status
        }
        trial_data.update(trial.hyperparameters.values)
        results_data.append(trial_data)
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('score', ascending=False)
    
    # Save to CSV
    results_path = os.path.join(save_dir, 'tuning_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hps_path = os.path.join(save_dir, 'best_hyperparameters.txt')
    
    with open(best_hps_path, 'w') as f:
        f.write("Best Hyperparameters:\n")
        f.write("=" * 50 + "\n")
        for param_name, param_value in best_hps.values.items():
            f.write(f"{param_name}: {param_value}\n")
    
    print(f"Tuning results saved to {save_dir}")
    print(f"Results CSV: {results_path}")
    print(f"Best hyperparameters: {best_hps_path}")

def main():
    """
    Test the enhanced model creation.
    """
    # Example shapes
    time_series_shape = (3000,)
    feature_shape = (20,)
    
    # Create enhanced model with default parameters
    model = create_enhanced_dual_input_model(time_series_shape, feature_shape)
    
    print("Enhanced model created successfully!")
    print(f"Time series input shape: {time_series_shape}")
    print(f"Feature input shape: {feature_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    model.summary()

if __name__ == "__main__":
    main()