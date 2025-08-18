"""
실험 설정 파일 - 어텐션 모드 및 모델 크기 비교 실험용
6800개 데이터셋에 최적화된 모델 크기 설정
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path
from models.property_prediction_transformer import PropertyPredictionConfig

@dataclass
class ExperimentConfig:
    """실험 설정 클래스"""
    name: str
    model_type: str  # "property" or "decision"
    attention_mode: str  # "standard" or "advanced"
    model_size: str  # "small", "medium", "large"
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    learning_rate: float
    batch_size: int
    num_epochs: int
    save_name: str  # 모델 저장시 사용할 이름

# 6800개 데이터에 맞는 모델 크기 정의 (과적합 방지)
MODEL_SIZES = {
    "small": {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 3,
        "d_ff": 512,
        "dropout": 0.2,
        "batch_size": 32
    },
    "medium": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 1024,
        "dropout": 0.15,
        "batch_size": 16
    },
    "large": {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "batch_size": 8
    }
}

def create_experiment_configs() -> List[ExperimentConfig]:
    """모든 실험 설정 생성"""
    configs = []
    
    # 1. Property 모델 - 크기별 비교 (Advanced 어텐션만)
    for size in ["small", "medium", "large"]:
        size_config = MODEL_SIZES[size]
        config = ExperimentConfig(
            name=f"property_advanced_{size}",
            model_type="property",
            attention_mode="advanced",
            model_size=size,
            d_model=size_config["d_model"],
            n_heads=size_config["n_heads"],
            n_layers=size_config["n_layers"],
            d_ff=size_config["d_ff"],
            dropout=size_config["dropout"],
            learning_rate=1e-2,
            batch_size=64,
            num_epochs=100,
            save_name=f"property_adv_{size}_{size_config['d_model']}d_{size_config['n_layers']}l"
        )
        configs.append(config)
    
    # 2. Decision 모델 - 어텐션 모드 비교 (Medium 크기)
    medium_config = MODEL_SIZES["medium"]
    for attention_mode in ["standard", "advanced"]:
        config = ExperimentConfig(
            name=f"decision_{attention_mode}_medium",
            model_type="decision",
            attention_mode=attention_mode,
            model_size="medium",
            d_model=medium_config["d_model"],
            n_heads=medium_config["n_heads"],
            n_layers=medium_config["n_layers"],
            d_ff=medium_config["d_ff"],
            dropout=medium_config["dropout"],
            learning_rate=1e-2,
            batch_size=medium_config["batch_size"],
            num_epochs=100,
            save_name=f"decision_{attention_mode}_med_{medium_config['d_model']}d_{medium_config['n_layers']}l"
        )
        configs.append(config)
    
    # 3. Property 모델 - 어텐션 모드 비교 (Medium 크기)
    for attention_mode in ["standard", "advanced"]:
        config = ExperimentConfig(
            name=f"property_{attention_mode}_medium",
            model_type="property",
            attention_mode=attention_mode,
            model_size="medium",
            d_model=medium_config["d_model"],
            n_heads=medium_config["n_heads"],
            n_layers=medium_config["n_layers"],
            d_ff=medium_config["d_ff"],
            dropout=medium_config["dropout"],
            learning_rate=1e-2,
            batch_size=64,
            num_epochs=100,
            save_name=f"property_{attention_mode}_med_{medium_config['d_model']}d_{medium_config['n_layers']}l"
        )
        configs.append(config)
    
    return configs


def create_property_prediction_config(size: str = "medium", attention_mode: str = "standard") -> PropertyPredictionConfig:
    """Property 모델 설정 생성 (RTG Calculator용)"""
    
    size_config = MODEL_SIZES[size]
    
    return PropertyPredictionConfig(
        d_model=size_config["d_model"],
        n_heads=size_config["n_heads"],
        n_layers=size_config["n_layers"],
        d_ff=size_config["d_ff"],
        dropout=size_config["dropout"],
        attention_mode=attention_mode,
        max_seq_len=512,
        vocab_size=100,  # 기본값
        num_properties=3  # entanglement, fidelity, expressibility
    )


def get_property_checkpoint_path(size: str = "medium", attention_mode: str = "standard") -> str:
    """Property 모델 체크포인트 경로 반환"""
    size_config = MODEL_SIZES[size]
    checkpoint_name = f"property_{attention_mode}_{size}_{size_config['d_model']}d_{size_config['n_layers']}l"
    return f"property_prediction_checkpoints/{checkpoint_name}_best_model.pt"

def get_experiment_config(experiment_name: str) -> ExperimentConfig:
    """특정 실험 설정 가져오기"""
    configs = create_experiment_configs()
    for config in configs:
        if config.name == experiment_name:
            return config
    raise ValueError(f"실험 설정을 찾을 수 없습니다: {experiment_name}")

def list_experiments() -> List[str]:
    """사용 가능한 실험 목록 반환"""
    configs = create_experiment_configs()
    return [config.name for config in configs]

def save_experiment_configs(save_path: str = "experiment_configs.json"):
    """실험 설정을 JSON 파일로 저장"""
    configs = create_experiment_configs()
    config_dict = {}
    
    for config in configs:
        config_dict[config.name] = {
            "model_type": config.model_type,
            "attention_mode": config.attention_mode,
            "model_size": config.model_size,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "save_name": config.save_name
        }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"📁 실험 설정 저장 완료: {save_path}")

if __name__ == "__main__":
    # 실험 설정 출력
    configs = create_experiment_configs()
    print("🔬 생성된 실험 설정:")
    print("\n📊 Property 모델 크기 비교 (Advanced 어텐션):")
    for config in configs:
        if config.model_type == "property" and config.attention_mode == "advanced":
            print(f"  - {config.name}: {config.d_model}d, {config.n_layers}l, {config.n_heads}h")
    
    print("\n🔄 어텐션 모드 비교 (Medium 크기):")
    for config in configs:
        if config.model_size == "medium":
            print(f"  - {config.name}: {config.attention_mode} attention")
    
    # JSON 파일로 저장
    save_experiment_configs()
