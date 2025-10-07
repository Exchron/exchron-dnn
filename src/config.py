import os

class Config:
    def __init__(self):
        # Data directories
        self.raw_data_dir = os.path.join("data", "raw")
        self.processed_data_dir = os.path.join("data", "processed")
        self.lightkurve_data_dir = os.path.join(self.raw_data_dir, "lightkurve_data")
        
        # Feature dataset
        self.feature_data_file = os.path.join(self.raw_data_dir, "KOI Selected 2000 Signals.csv")
        
        # Model parameters
        self.time_series_length = 3000  # Length of time series after preprocessing
        self.num_features = 20  # Number of extracted features
        self.num_classes = 2  # Binary classification with softmax output (2 classes)
        self.batch_size = 32
        self.epochs = 100
        self.validation_split = 0.2
        self.test_split = 0.15
        
        # DNN architecture parameters for time series branch
        self.conv_filters = [32, 64, 128]
        self.conv_kernel_sizes = [7, 5, 3]
        self.pool_sizes = [4, 4, 2]
        self.dropout_rates = [0.25, 0.3, 0.4, 0.5]
        
        # DNN architecture parameters for feature branch
        self.feature_dense_layers = [64, 32]
        
        # Combined model parameters
        self.combined_dense_layers = [128, 64, 32]
        self.activation_function = 'relu'
        self.output_activation = 'softmax'  # For two-class probability output
        
        # Enhanced model parameters
        self.use_enhanced_model = False
        self.use_attention = True
        self.use_multi_scale_cnn = True
        self.use_skip_connections = True
        self.fusion_strategy = 'concatenate'  # Options: 'concatenate', 'attention_fusion', 'bilinear'
        
        # Hyperparameter tuning parameters
        self.enable_hyperparameter_tuning = True
        self.max_tuning_trials = 30
        self.tuner_type = 'bayesian'  # Options: 'random', 'bayesian', 'hyperband'
        self.tuning_objective = 'val_auc'
        self.executions_per_trial = 1
        
        # Training parameters
        self.learning_rate = 0.001
        self.early_stopping_patience = 15
        self.reduce_lr_patience = 10
        self.reduce_lr_factor = 0.5
        
        # Enhanced training parameters
        self.use_class_weights = True
        self.use_focal_loss = False
        self.gradient_clipping = 1.0
        self.use_cosine_annealing = False
        self.warmup_epochs = 5
        
        # Data preprocessing
        self.normalize_flux = True
        self.remove_outliers = True
        self.outlier_threshold = 3.0  # Standard deviations
        
        # SHAP configuration
        self.shap_explainer_type = 'DeepExplainer'
        self.shap_background_samples = 100
        
        # Logging and saving
        self.model_save_path = os.path.join("models", "dual_input_dnn_model.keras")
        self.log_dir = os.path.join("logs")
        self.checkpoint_path = os.path.join("models", "checkpoints")
        
        # Other configurations
        self.random_seed = 42  # For reproducibility
        self.verbose = 1

config = Config()