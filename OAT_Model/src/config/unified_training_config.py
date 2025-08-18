"""
Unified Training Configuration Management
Centralized hyperparameter management for all models
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch


@dataclass
class ModelArchitectureConfig:
    """Model architecture hyperparameters"""
    # Shared architecture parameters
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Unified attention configuration
    attention_mode: str = "advanced"  # "standard", "advanced", "grid", "semantic"
    use_rotary_pe: bool = True
    
    # Gate and circuit parameters
    max_qubits: int = 50
    max_gates: int = 300
    n_gate_types: Optional[int] = None  # Auto-detected from gate registry
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def get_device(self) -> str:
        """Get actual device string"""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Scheduler settings
    scheduler_type: str = "cosine"  # "cosine", "onecycle", "linear"
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # Training loop settings
    num_epochs: int = 100
    train_batch_size: int = 32
    val_batch_size: int = 32
    test_batch_size: int = 32
    
    # Gradient settings
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    
    # Validation and saving
    val_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    early_stopping_patience: int = 999999  # Effectively disable early stopping
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "quantum-transformer"


@dataclass
class DecisionTransformerConfig:
    """Decision Transformer specific configuration"""
    # Generation settings
    max_generation_length: int = 50
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.9
    do_sample: bool = True
    
    # Reward settings
    use_reward_guidance: bool = True
    reward_weight: float = 1.0
    
    # Architecture specific (inherits from ModelArchitectureConfig)
    position_dim: Optional[int] = None  # Auto-detected from checkpoint


@dataclass
class PropertyPredictorConfig:
    """Property Predictor specific configuration"""
    # Output settings
    property_dim: int = 3  # entanglement, fidelity, expressibility
    
    # Loss weights (rebalanced for convergence)
    entanglement_weight: float = 10.0
    fidelity_weight: float = 100.0
    expressibility_weight: float = 1.0
    combined_weight: float = 0.5
    
    # Architecture specific (inherits from ModelArchitectureConfig)
    use_global_pooling: bool = True  # Use global pooling for circuit representation


@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset paths
    data_path: str = "dummy_experiment_results.json"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Data processing
    max_circuit_length: int = 50
    normalize_targets: bool = True
    augment_data: bool = False
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "cache"


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    tags: list = field(default_factory=list)
    
    # Output directories
    output_dir: str = "experiments"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class UnifiedTrainingConfig:
    """통합 학습 설정"""
    
    # 모델 설정
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    
    # 학습 설정
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 데이터 설정
    data: DataConfig = field(default_factory=DataConfig)

    # RTG 설정
    enable_rtg: bool = False
    property_model_size: str = "medium"
    property_attention_mode: str = "standard"
    
    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'UnifiedTrainingConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        config = cls()
        
        if 'model' in data:
            config.model = ModelArchitectureConfig(**data['model'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'decision_transformer' in data:
            config.decision_transformer = DecisionTransformerConfig(**data['decision_transformer'])
        if 'property_predictor' in data:
            config.property_predictor = PropertyPredictorConfig(**data['property_predictor'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'experiment' in data:
            config.experiment = ExperimentConfig(**data['experiment'])
        
        return config
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in updates.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def get_model_config_for_decision_transformer(self) -> Dict[str, Any]:
        """Get model configuration for Decision Transformer"""
        return {
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_layers': self.model.n_layers,
            'd_ff': self.model.d_ff,
            'dropout': self.model.dropout,
            'max_qubits': self.model.max_qubits,
            'n_gate_types': self.model.n_gate_types,
            'position_dim': self.decision_transformer.position_dim,
            'attention_mode': self.decision_transformer.attention_mode,
            'device': self.model.get_device()
        }
    
    def get_model_config_for_property_predictor(self) -> Dict[str, Any]:
        """Get model configuration for Property Predictor"""
        return {
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_layers': self.model.n_layers,
            'd_ff': self.model.d_ff,
            'dropout': self.model.dropout,
            'max_qubits': self.model.max_qubits,
            'max_gates': self.model.max_gates,
            'device': self.model.get_device()
        }
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.experiment.output_dir,
            self.experiment.checkpoint_dir,
            self.experiment.log_dir,
            self.data.cache_dir
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def set_seed(self):
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        
        random.seed(self.experiment.seed)
        np.random.seed(self.experiment.seed)
        torch.manual_seed(self.experiment.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.experiment.seed)
            torch.cuda.manual_seed_all(self.experiment.seed)
        
        if self.experiment.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Predefined experiment configurations
def get_small_experiment_config() -> UnifiedTrainingConfig:
    """Small experiment for quick testing"""
    config = UnifiedTrainingConfig()
    
    # Small model
    config.model.d_model = 256
    config.model.n_layers = 4
    config.model.n_heads = 4
    config.model.d_ff = 1024
    
    # Fast training
    config.training.num_epochs = 10
    config.training.train_batch_size = 16
    config.training.val_every_n_steps = 100
    config.training.save_every_n_steps = 200
    
    config.experiment.experiment_name = "small_test"
    config.experiment.description = "Small model for quick testing"
    
    return config


def get_medium_experiment_config() -> UnifiedTrainingConfig:
    """Medium experiment for development"""
    config = UnifiedTrainingConfig()
    
    # Medium model (default values are already medium)
    config.experiment.experiment_name = "medium_dev"
    config.experiment.description = "Medium model for development"
    
    return config


def get_large_experiment_config() -> UnifiedTrainingConfig:
    """Large experiment for production"""
    config = UnifiedTrainingConfig()
    
    # Large model
    config.model.d_model = 768
    config.model.n_layers = 12
    config.model.n_heads = 12
    config.model.d_ff = 3072
    config.model.max_qubits = 16
    
    # Intensive training
    config.training.num_epochs = 200
    config.training.train_batch_size = 64
    config.training.learning_rate = 5e-5
    config.training.warmup_steps = 2000
    
    config.experiment.experiment_name = "large_production"
    config.experiment.description = "Large model for production use"
    
    return config


def get_config_by_name(name: str) -> UnifiedTrainingConfig:
    """Get predefined configuration by name"""
    configs = {
        'small': get_small_experiment_config,
        'medium': get_medium_experiment_config,
        'large': get_large_experiment_config
    }
    
    if name not in configs:
        raise ValueError(f"Unknown config name: {name}. Available: {list(configs.keys())}")
    
    return configs[name]()


# Configuration manager class
class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: UnifiedTrainingConfig, name: str):
        """Save configuration with a name"""
        config_path = self.config_dir / f"{name}.json"
        config.save(config_path)
        print(f"Configuration saved to {config_path}")
    
    def load_config(self, name: str) -> UnifiedTrainingConfig:
        """Load configuration by name"""
        config_path = self.config_dir / f"{name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        return UnifiedTrainingConfig.load(config_path)
    
    def list_configs(self) -> list:
        """List available configuration files"""
        return [f.stem for f in self.config_dir.glob("*.json")]
    
    def create_experiment_config(self, 
                               base_config: str = "medium",
                               experiment_name: str = None,
                               overrides: Dict[str, Any] = None) -> UnifiedTrainingConfig:
        """Create experiment configuration with overrides"""
        # Get base configuration
        config = get_config_by_name(base_config)
        
        # Set experiment name
        if experiment_name:
            config.experiment.experiment_name = experiment_name
        
        # Apply overrides
        if overrides:
            config.update_from_dict(overrides)
        
        # Setup experiment
        config.setup_directories()
        config.set_seed()
        
        return config


if __name__ == "__main__":
    # Example usage
    print("🔧 Unified Training Configuration System")
    
    # Create config manager
    manager = ConfigManager()
    
    # Create and save different experiment configs
    small_config = get_small_experiment_config()
    manager.save_config(small_config, "small_test")
    
    medium_config = get_medium_experiment_config()
    manager.save_config(medium_config, "medium_dev")
    
    large_config = get_large_experiment_config()
    manager.save_config(large_config, "large_production")
    
    print(f"Available configurations: {manager.list_configs()}")
    
    # Example of creating custom experiment
    custom_config = manager.create_experiment_config(
        base_config="medium",
        experiment_name="custom_experiment",
        overrides={
            "model": {"d_model": 384, "n_layers": 8},
            "training": {"learning_rate": 2e-4, "num_epochs": 50}
        }
    )
    
    print(f"Custom experiment created: {custom_config.experiment.experiment_name}")
    print(f"Model d_model: {custom_config.model.d_model}")
    print(f"Training epochs: {custom_config.training.num_epochs}")
