"""
Optimized Property Prediction Configuration
과적합 문제 해결을 위한 최적화된 설정
"""

from models.property_prediction_transformer import PropertyPredictionConfig, PropertyPredictionLoss
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def create_optimized_config():
    """과적합 해결을 위한 최적화된 설정"""
    return PropertyPredictionConfig(
        # 🔧 모델 크기 대폭 축소 (과적합 방지)
        d_model=256,        # 512 → 256 (50% 감소)
        n_heads=8,          # 유지 (적절한 어텐션 헤드)
        n_layers=4,         # 6 → 4 (33% 감소)
        d_ff=1024,          # 2048 → 1024 (50% 감소)
        
        # 🛡️ 정규화 강화
        dropout=0.4,        # 0.3 → 0.4 (과적합 방지)
        weight_decay=1e-2,  # 1e-3 → 1e-2 (10배 강화)
        
        # ⚡ 학습 최적화
        learning_rate=5e-4, # 1e-4 → 5e-4 (빠른 수렴)
        warmup_steps=500,   # 1000 → 500 (빠른 워밍업)
        
        # 📊 배치 설정
        train_batch_size=64,  # 안정적인 배치 크기
        val_batch_size=64,
        
        # 🎯 출력 설정 (robust_fidelity 제거됨)
        property_dim=3  # entanglement, fidelity, expressibility
    )


def create_optimized_loss():
    """최적화된 손실 함수"""
    return PropertyPredictionLoss(
        entanglement_weight=1.0,
        fidelity_weight=10.0,     # 5.0 → 10.0 (중요도 증가)
        expressibility_weight=0.05, # 0.1 → 0.05 (큰 값이므로 가중치 감소)
        combined_weight=0.3       # 0.5 → 0.3 (개별 헤드 중심)
    )


def create_optimized_optimizer(model, config):
    """최적화된 옵티마이저"""
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )


def create_optimized_scheduler(optimizer):
    """최적화된 스케줄러 (적응적)"""
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # 학습률 50% 감소
        patience=3,      # 3 에포크 대기 (빠른 반응)
        verbose=True,
        min_lr=1e-6,     # 최소 학습률
        threshold=1e-3   # 개선 임계값
    )


class OptimizedTrainingConfig:
    """최적화된 훈련 설정"""
    
    # 그래디언트 클리핑
    GRADIENT_CLIP_NORM = 1.0
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10  # 15 → 10 (빠른 중단)
    
    # 체크포인트 저장
    SAVE_BEST_MODEL = True
    SAVE_EVERY_N_EPOCHS = 5
    
    # 디버깅
    LOG_EVERY_N_STEPS = 10
    DETAILED_LOG_EVERY_N_EPOCHS = 3
    
    # 검증
    VALIDATE_EVERY_EPOCH = True


def print_optimization_summary():
    """최적화 요약 출력"""
    print("🚀 Property Prediction 최적화 설정")
    print("=" * 50)
    
    print("\n📉 모델 크기 축소:")
    print("   d_model: 512 → 256 (50% 감소)")
    print("   n_layers: 6 → 4 (33% 감소)")
    print("   d_ff: 2048 → 1024 (50% 감소)")
    print("   예상 파라미터: ~15M (기존 58M에서 74% 감소)")
    
    print("\n🛡️ 과적합 방지:")
    print("   dropout: 0.3 → 0.4")
    print("   weight_decay: 1e-3 → 1e-2 (10배 강화)")
    print("   그래디언트 클리핑: 1.0")
    
    print("\n⚡ 학습 가속:")
    print("   learning_rate: 1e-4 → 5e-4 (5배 증가)")
    print("   스케줄러: ReduceLROnPlateau (patience=3)")
    print("   Early stopping: patience=10")
    
    print("\n🎯 손실 함수 재조정:")
    print("   Fidelity 가중치: 5.0 → 10.0 (중요도 증가)")
    print("   Expressibility 가중치: 0.1 → 0.05 (큰 값 보정)")
    
    print("\n✅ 기대 효과:")
    print("   1. 과적합 해결 → 검증 손실 개선")
    print("   2. 빠른 수렴 → 1.5 정체 돌파")
    print("   3. 안정적인 학습 → 그래디언트 안정성")
    print("   4. 메모리 효율성 → 74% 파라미터 감소")


if __name__ == "__main__":
    print_optimization_summary()
    
    # 설정 생성 테스트
    config = create_optimized_config()
    print(f"\n🔧 생성된 최적화 설정:")
    print(f"   d_model: {config.d_model}")
    print(f"   n_layers: {config.n_layers}")
    print(f"   dropout: {config.dropout}")
    print(f"   learning_rate: {config.learning_rate}")
    print(f"   weight_decay: {config.weight_decay}")
