import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_visualizations(model, history, results, test_data, feature_names, 
                                 feature_scaler, config, save_dir='enhanced_visualizations',
                                 best_hps=None):
    """
    Create comprehensive enhanced visualizations for the exoplanet classification model.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    history : History
        Training history
    results : dict
        Evaluation results
    test_data : tuple
        Test dataset (sequences, features, labels)
    feature_names : list
        Names of features
    feature_scaler : sklearn.preprocessing.StandardScaler
        Feature scaler used during preprocessing
    config : Config
        Configuration object
    save_dir : str
        Directory to save visualizations
    best_hps : dict, optional
        Best hyperparameters from tuning
    """
    # Create directory structure
    os.makedirs(save_dir, exist_ok=True)
    subdirs = ['performance', 'feature_analysis', 'model_analysis', 'data_analysis', 'predictions']
    for subdir in subdirs:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
    
    test_sequences, test_features, test_labels = test_data
    predicted_probs = results['predicted_probs']
    predicted_classes = results['predicted_classes']
    
    print("Creating enhanced visualizations...")
    
    # 1. Enhanced Performance Visualizations
    create_performance_dashboard(history, results, test_labels, predicted_probs, 
                               os.path.join(save_dir, 'performance'))
    
    # 2. Feature Analysis
    create_feature_analysis(test_features, feature_names, test_labels, feature_scaler,
                           os.path.join(save_dir, 'feature_analysis'))
    
    # 3. Model Analysis
    create_model_analysis(model, test_data, history, 
                         os.path.join(save_dir, 'model_analysis'))
    
    # 4. Data Analysis
    create_data_analysis(test_sequences, test_labels, 
                        os.path.join(save_dir, 'data_analysis'))
    
    # 5. Prediction Analysis
    create_prediction_analysis(test_labels, predicted_probs, predicted_classes,
                              os.path.join(save_dir, 'predictions'))
    
    # 6. Create comprehensive report
    create_analysis_report(results, config, save_dir, best_hps)
    
    print(f"✓ Enhanced visualizations created in: {save_dir}")

def create_performance_dashboard(history, results, test_labels, predicted_probs, save_dir):
    """Create an interactive performance dashboard."""
    
    # 1. Training History Dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Training/Validation Loss', 'Training/Validation Accuracy',
                       'Training/Validation Precision', 'Training/Validation Recall',
                       'ROC Curve', 'Precision-Recall Curve'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Loss
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['loss'], 
                            name='Training Loss', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_loss'], 
                            name='Validation Loss', line=dict(color='blue')), row=1, col=1)
    
    # Accuracy
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['accuracy'], 
                            name='Training Accuracy', line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_accuracy'], 
                            name='Validation Accuracy', line=dict(color='blue')), row=1, col=2)
    
    # Precision
    if 'precision' in history.history:
        fig.add_trace(go.Scatter(x=list(epochs), y=history.history['precision'], 
                                name='Training Precision', line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_precision'], 
                                name='Validation Precision', line=dict(color='blue')), row=2, col=1)
    
    # Recall
    if 'recall' in history.history:
        fig.add_trace(go.Scatter(x=list(epochs), y=history.history['recall'], 
                                name='Training Recall', line=dict(color='red')), row=2, col=2)
        fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_recall'], 
                                name='Validation Recall', line=dict(color='blue')), row=2, col=2)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, predicted_probs)
    auc_score = results['auc_score']
    
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={auc_score:.3f})', 
                            line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', 
                            line=dict(dash='dash', color='gray')), row=3, col=1)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(test_labels, predicted_probs)
    fig.add_trace(go.Scatter(x=recall, y=precision, name='PR Curve', 
                            line=dict(color='orange')), row=3, col=2)
    
    fig.update_layout(height=900, title_text="Model Performance Dashboard", showlegend=False)
    fig.write_html(os.path.join(save_dir, 'performance_dashboard.html'))
    
    # 2. Static Performance Summary
    plt.figure(figsize=(20, 15))
    
    # Training history
    plt.subplot(3, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='red')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
    plt.title('Training/Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # ROC Curve
    plt.subplot(3, 3, 3)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', color='green', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    # Confusion Matrix
    plt.subplot(3, 3, 4)
    cm = confusion_matrix(test_labels, results['predicted_classes'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['False Positive', 'Candidate'],
                yticklabels=['False Positive', 'Candidate'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Precision-Recall Curve
    plt.subplot(3, 3, 5)
    plt.plot(recall, precision, color='orange', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    # Prediction Distribution
    plt.subplot(3, 3, 6)
    plt.hist(predicted_probs[test_labels == 0], bins=50, alpha=0.7, label='False Positives', color='red')
    plt.hist(predicted_probs[test_labels == 1], bins=50, alpha=0.7, label='Candidates', color='green')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate (if available)
    if hasattr(history, 'lr') or 'lr' in history.history:
        plt.subplot(3, 3, 7)
        lr_values = history.history.get('lr', [])
        if lr_values:
            plt.plot(lr_values)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
    
    # Metrics comparison
    plt.subplot(3, 3, 8)
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    values = [results['test_accuracy'], results['test_precision'], 
              results['test_recall'], results['auc_score']]
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    bars = plt.bar(metrics, values, color=colors)
    plt.title('Test Metrics Summary')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_analysis(test_features, feature_names, test_labels, feature_scaler, save_dir):
    """Create comprehensive feature analysis visualizations."""
    
    # 1. Feature importance using permutation importance
    try:
        from sklearn.inspection import permutation_importance
        # Note: This would require the model, skipping for now
        pass
    except:
        pass
    
    # 2. Feature distributions
    plt.figure(figsize=(20, 15))
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    for i, feature_name in enumerate(feature_names):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot distributions for both classes
        false_pos_values = test_features[test_labels == 0, i]
        candidate_values = test_features[test_labels == 1, i]
        
        plt.hist(false_pos_values, bins=30, alpha=0.7, label='False Positives', 
                color='red', density=True)
        plt.hist(candidate_values, bins=30, alpha=0.7, label='Candidates', 
                color='green', density=True)
        
        plt.title(f'{feature_name}')
        plt.xlabel('Normalized Value')
        plt.ylabel('Density')
        if i == 0:
            plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = np.corrcoef(test_features.T)
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. PCA analysis
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(test_features)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                         c=test_labels, cmap='RdYlGn', alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA of Features')
    plt.colorbar(scatter, label='Class (0=FP, 1=Candidate)')
    plt.grid(True, alpha=0.3)
    
    # PCA component contributions
    plt.subplot(1, 2, 2)
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    components_df.plot(kind='barh', ax=plt.gca())
    plt.title('PCA Component Contributions')
    plt.xlabel('Component Weight')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_model_analysis(model, test_data, history, save_dir):
    """Create model architecture and behavior analysis."""
    
    test_sequences, test_features, test_labels = test_data
    
    # 1. Model architecture visualization
    try:
        tf.keras.utils.plot_model(
            model, 
            to_file=os.path.join(save_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=150
        )
    except:
        print("Could not create model architecture plot (graphviz might be missing)")
    
    # 2. Layer analysis
    layer_info = []
    total_params = 0
    
    for layer in model.layers:
        if hasattr(layer, 'count_params'):
            params = layer.count_params()
            total_params += params
            layer_info.append({
                'Layer': layer.name,
                'Type': layer.__class__.__name__,
                'Output Shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                'Parameters': params
            })
    
    # Save layer information
    layer_df = pd.DataFrame(layer_info)
    layer_df.to_csv(os.path.join(save_dir, 'layer_analysis.csv'), index=False)
    
    # 3. Training convergence analysis
    plt.figure(figsize=(15, 10))
    
    # Loss convergence
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Overfitting analysis
    plt.subplot(2, 3, 2)
    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    overfitting_gap = val_loss - train_loss
    
    plt.plot(overfitting_gap, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Overfitting Gap (Val - Train Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.grid(True)
    
    # Learning curves smoothed
    plt.subplot(2, 3, 3)
    window = min(10, len(history.history['loss']) // 10)
    if window > 1:
        smooth_train_loss = pd.Series(history.history['loss']).rolling(window).mean()
        smooth_val_loss = pd.Series(history.history['val_loss']).rolling(window).mean()
        plt.plot(smooth_train_loss, label='Smoothed Training Loss')
        plt.plot(smooth_val_loss, label='Smoothed Validation Loss')
        plt.title(f'Smoothed Learning Curves (window={window})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # Parameter distribution
    plt.subplot(2, 3, 4)
    param_counts = [info['Parameters'] for info in layer_info if info['Parameters'] > 0]
    layer_names = [info['Layer'] for info in layer_info if info['Parameters'] > 0]
    
    if param_counts:
        plt.barh(layer_names, param_counts)
        plt.title('Parameters per Layer')
        plt.xlabel('Number of Parameters')
        plt.grid(True, alpha=0.3)
    
    # Model complexity metrics
    plt.subplot(2, 3, 5)
    metrics = ['Total Layers', 'Trainable Params', 'Non-trainable Params']
    values = [len(model.layers), model.count_params(), 
              sum([layer.count_params() for layer in model.layers if not layer.trainable])]
    
    plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Complexity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Training efficiency
    plt.subplot(2, 3, 6)
    if len(history.history['loss']) > 1:
        loss_improvement = np.diff(history.history['val_loss'])
        plt.plot(loss_improvement, marker='o', markersize=3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.title('Validation Loss Improvement per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_data_analysis(test_sequences, test_labels, save_dir):
    """Create data analysis visualizations."""
    
    # 1. Light curve examples
    plt.figure(figsize=(20, 12))
    
    # Show examples of both classes
    fp_indices = np.where(test_labels == 0)[0][:6]
    candidate_indices = np.where(test_labels == 1)[0][:6]
    
    for i, idx in enumerate(fp_indices):
        plt.subplot(3, 4, i + 1)
        plt.plot(test_sequences[idx], color='red', alpha=0.7)
        plt.title(f'False Positive {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Flux')
        plt.grid(True, alpha=0.3)
    
    for i, idx in enumerate(candidate_indices):
        plt.subplot(3, 4, i + 7)
        plt.plot(test_sequences[idx], color='green', alpha=0.7)
        plt.title(f'Candidate {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Flux')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lightcurve_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical analysis of light curves
    plt.figure(figsize=(15, 10))
    
    # Calculate statistics for each class
    fp_sequences = test_sequences[test_labels == 0]
    candidate_sequences = test_sequences[test_labels == 1]
    
    # Mean light curves
    plt.subplot(2, 3, 1)
    fp_mean = np.mean(fp_sequences, axis=0)
    candidate_mean = np.mean(candidate_sequences, axis=0)
    
    plt.plot(fp_mean, label='False Positives Mean', color='red', alpha=0.8)
    plt.plot(candidate_mean, label='Candidates Mean', color='green', alpha=0.8)
    plt.title('Mean Light Curves by Class')
    plt.xlabel('Time')
    plt.ylabel('Normalized Flux')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Standard deviation
    plt.subplot(2, 3, 2)
    fp_std = np.std(fp_sequences, axis=0)
    candidate_std = np.std(candidate_sequences, axis=0)
    
    plt.plot(fp_std, label='False Positives Std', color='red', alpha=0.8)
    plt.plot(candidate_std, label='Candidates Std', color='green', alpha=0.8)
    plt.title('Standard Deviation by Class')
    plt.xlabel('Time')
    plt.ylabel('Std Dev')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Variance distribution
    plt.subplot(2, 3, 3)
    fp_var = np.var(fp_sequences, axis=1)
    candidate_var = np.var(candidate_sequences, axis=1)
    
    plt.hist(fp_var, bins=30, alpha=0.7, label='False Positives', color='red', density=True)
    plt.hist(candidate_var, bins=30, alpha=0.7, label='Candidates', color='green', density=True)
    plt.title('Variance Distribution')
    plt.xlabel('Variance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Min/Max distribution
    plt.subplot(2, 3, 4)
    fp_range = np.max(fp_sequences, axis=1) - np.min(fp_sequences, axis=1)
    candidate_range = np.max(candidate_sequences, axis=1) - np.min(candidate_sequences, axis=1)
    
    plt.hist(fp_range, bins=30, alpha=0.7, label='False Positives', color='red', density=True)
    plt.hist(candidate_range, bins=30, alpha=0.7, label='Candidates', color='green', density=True)
    plt.title('Range Distribution')
    plt.xlabel('Max - Min')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Power spectrum analysis (simplified)
    plt.subplot(2, 3, 5)
    # Calculate mean power for each class
    fp_power = np.mean([np.abs(np.fft.fft(seq)[:len(seq)//2]) for seq in fp_sequences[:100]], axis=0)
    candidate_power = np.mean([np.abs(np.fft.fft(seq)[:len(seq)//2]) for seq in candidate_sequences[:100]], axis=0)
    
    freqs = np.fft.fftfreq(test_sequences.shape[1])[:test_sequences.shape[1]//2]
    
    plt.plot(freqs[:100], fp_power[:100], label='False Positives', color='red', alpha=0.8)
    plt.plot(freqs[:100], candidate_power[:100], label='Candidates', color='green', alpha=0.8)
    plt.title('Mean Power Spectrum (Low Freq)')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Class balance
    plt.subplot(2, 3, 6)
    class_counts = [len(fp_sequences), len(candidate_sequences)]
    class_labels = ['False Positives', 'Candidates']
    colors = ['red', 'green']
    
    plt.pie(class_counts, labels=class_labels, colors=colors, autopct='%1.1f%%')
    plt.title('Class Distribution in Test Set')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_analysis(test_labels, predicted_probs, predicted_classes, save_dir):
    """Create prediction analysis visualizations."""
    
    plt.figure(figsize=(15, 10))
    
    # 1. Prediction confidence distribution
    plt.subplot(2, 3, 1)
    plt.hist(predicted_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Confidence by true class
    plt.subplot(2, 3, 2)
    fp_probs = predicted_probs[test_labels == 0]
    candidate_probs = predicted_probs[test_labels == 1]
    
    plt.hist(fp_probs, bins=30, alpha=0.7, label='False Positives', color='red', density=True)
    plt.hist(candidate_probs, bins=30, alpha=0.7, label='Candidates', color='green', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    plt.title('Confidence by True Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Calibration plot
    plt.subplot(2, 3, 3)
    # Bin predictions and calculate actual vs predicted probabilities
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = test_labels[in_bin].mean()
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_centers, bin_accuracies, 'o-', label='Model Calibration', markersize=8)
    plt.title('Calibration Plot')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Error analysis
    plt.subplot(2, 3, 4)
    # False positives and false negatives
    fp_errors = (predicted_classes == 1) & (test_labels == 0)  # Predicted positive, actually negative
    fn_errors = (predicted_classes == 0) & (test_labels == 1)  # Predicted negative, actually positive
    
    fp_confidences = predicted_probs[fp_errors]
    fn_confidences = predicted_probs[fn_errors]
    
    if len(fp_confidences) > 0:
        plt.hist(fp_confidences, bins=20, alpha=0.7, label=f'False Positives (n={len(fp_confidences)})', 
                color='red', density=True)
    if len(fn_confidences) > 0:
        plt.hist(fn_confidences, bins=20, alpha=0.7, label=f'False Negatives (n={len(fn_confidences)})', 
                color='orange', density=True)
    
    plt.title('Error Analysis by Confidence')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Threshold analysis
    plt.subplot(2, 3, 5)
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        pred_at_threshold = (predicted_probs >= threshold).astype(int)
        
        if np.sum(pred_at_threshold) > 0:
            precision = np.sum((pred_at_threshold == 1) & (test_labels == 1)) / np.sum(pred_at_threshold == 1)
            recall = np.sum((pred_at_threshold == 1) & (test_labels == 1)) / np.sum(test_labels == 1)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = 0
            recall = 0
            f1 = 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='red')
    plt.plot(thresholds, f1_scores, label='F1 Score', color='green')
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')
    
    # Find optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.axvline(x=optimal_threshold, color='purple', linestyle=':', 
                label=f'Optimal F1 Threshold ({optimal_threshold:.3f})')
    
    plt.title('Threshold Analysis')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Prediction uncertainty
    plt.subplot(2, 3, 6)
    # Calculate uncertainty as distance from 0.5
    uncertainty = np.abs(predicted_probs - 0.5)
    
    # Separate by correctness
    correct_predictions = (predicted_classes == test_labels)
    correct_uncertainty = uncertainty[correct_predictions]
    incorrect_uncertainty = uncertainty[~correct_predictions]
    
    plt.hist(correct_uncertainty, bins=25, alpha=0.7, label=f'Correct (n={len(correct_uncertainty)})', 
             color='green', density=True)
    plt.hist(incorrect_uncertainty, bins=25, alpha=0.7, label=f'Incorrect (n={len(incorrect_uncertainty)})', 
             color='red', density=True)
    
    plt.title('Prediction Uncertainty')
    plt.xlabel('Distance from 0.5')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed error analysis
    error_analysis = {
        'total_samples': len(test_labels),
        'true_positives': np.sum((predicted_classes == 1) & (test_labels == 1)),
        'false_positives': np.sum((predicted_classes == 1) & (test_labels == 0)),
        'true_negatives': np.sum((predicted_classes == 0) & (test_labels == 0)),
        'false_negatives': np.sum((predicted_classes == 0) & (test_labels == 1)),
        'optimal_f1_threshold': optimal_threshold,
        'optimal_f1_score': f1_scores[optimal_idx]
    }
    
    pd.DataFrame([error_analysis]).to_csv(os.path.join(save_dir, 'error_analysis.csv'), index=False)

def create_analysis_report(results, config, save_dir, best_hps=None):
    """Create a comprehensive analysis report."""
    
    report_path = os.path.join(save_dir, 'ANALYSIS_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced Exoplanet Classification Analysis Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Performance Summary\n\n")
        f.write(f"- **Test Accuracy**: {results['test_accuracy']:.4f}\n")
        f.write(f"- **Test Precision**: {results['test_precision']:.4f}\n")
        f.write(f"- **Test Recall**: {results['test_recall']:.4f}\n")
        f.write(f"- **AUC Score**: {results['auc_score']:.4f}\n")
        f.write(f"- **Test Loss**: {results['test_loss']:.4f}\n\n")
        
        if best_hps:
            f.write("## Best Hyperparameters\n\n")
            for param_name, param_value in best_hps.items():
                f.write(f"- **{param_name}**: {param_value}\n")
            f.write("\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Time Series Length**: {config.time_series_length}\n")
        f.write(f"- **Batch Size**: {config.batch_size}\n")
        f.write(f"- **Learning Rate**: {config.learning_rate}\n")
        f.write(f"- **Epochs**: {config.epochs}\n")
        f.write(f"- **Random Seed**: {config.random_seed}\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("### Performance Analysis\n")
        f.write("- `performance/performance_dashboard.html`: Interactive performance dashboard\n")
        f.write("- `performance/performance_summary.png`: Static performance summary\n\n")
        
        f.write("### Feature Analysis\n")
        f.write("- `feature_analysis/feature_distributions.png`: Feature distributions by class\n")
        f.write("- `feature_analysis/feature_correlations.png`: Feature correlation matrix\n")
        f.write("- `feature_analysis/pca_analysis.png`: PCA analysis\n\n")
        
        f.write("### Model Analysis\n")
        f.write("- `model_analysis/model_architecture.png`: Model architecture visualization\n")
        f.write("- `model_analysis/model_analysis.png`: Model behavior analysis\n")
        f.write("- `model_analysis/layer_analysis.csv`: Layer-by-layer analysis\n\n")
        
        f.write("### Data Analysis\n")
        f.write("- `data_analysis/lightcurve_examples.png`: Example light curves\n")
        f.write("- `data_analysis/data_analysis.png`: Statistical data analysis\n\n")
        
        f.write("### Prediction Analysis\n")
        f.write("- `predictions/prediction_analysis.png`: Prediction analysis\n")
        f.write("- `predictions/error_analysis.csv`: Detailed error analysis\n\n")
        
        f.write("## Recommendations for Improvement\n\n")
        f.write("Based on the analysis, consider the following improvements:\n\n")
        f.write("1. **Data Augmentation**: If class imbalance is significant\n")
        f.write("2. **Feature Engineering**: Add more domain-specific features\n")
        f.write("3. **Ensemble Methods**: Combine multiple models\n")
        f.write("4. **Threshold Optimization**: Use optimal F1 threshold instead of 0.5\n")
        f.write("5. **Regularization**: Adjust based on overfitting analysis\n")
    
    print(f"✓ Analysis report saved to: {report_path}")

def main():
    """Test visualization creation."""
    print("Enhanced visualization module loaded successfully!")

if __name__ == "__main__":
    main()