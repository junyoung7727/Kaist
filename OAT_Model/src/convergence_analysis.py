"""
Property Prediction Convergence Analysis
수렴 문제의 핵심 원인 분석 및 해결책
"""

import torch
import torch.nn as nn
from models.property_prediction_transformer import PropertyPredictionConfig, PropertyPredictionTransformer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))


class ConvergenceAnalyzer:
    """수렴 문제 핵심 분석기"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_model_capacity(self, config: PropertyPredictionConfig):
        """모델 용량 분석"""
        model = PropertyPredictionTransformer(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"🧠 모델 용량 분석:")
        print(f"   d_model: {config.d_model}")
        print(f"   n_layers: {config.n_layers}")
        print(f"   총 파라미터: {total_params:,}")
        
        # 용량 문제 진단
        if total_params > 20_000_000:  # 20M 이상
            print(f"   ❌ 과적합 위험: 파라미터 수가 너무 많음")
            return "oversized"
        elif total_params < 1_000_000:  # 1M 미만
            print(f"   ❌ 용량 부족: 파라미터 수가 너무 적음")
            return "undersized"
        else:
            print(f"   ✅ 적절한 모델 크기")
            return "optimal"
    
    def recommend_optimal_config(self):
        """최적 설정 권장"""
        print(f"\n💡 수렴 개선을 위한 최적 설정:")
        
        # 작은 모델 설정 (과적합 방지)
        optimal_config = PropertyPredictionConfig(
            d_model=256,        # 512 -> 256 (50% 감소)
            n_heads=8,          # 유지
            n_layers=4,         # 6 -> 4 (33% 감소)
            d_ff=1024,          # 2048 -> 1024 (50% 감소)
            dropout=0.4,        # 0.3 -> 0.4 (과적합 방지)
            learning_rate=5e-4, # 1e-4 -> 5e-4 (학습 속도 증가)
            weight_decay=1e-2   # 1e-3 -> 1e-2 (정규화 강화)
        )
        
        capacity_status = self.analyze_model_capacity(optimal_config)
        
        print(f"\n🎯 권장 훈련 설정:")
        print(f"   학습률: {optimal_config.learning_rate} (초기값 증가)")
        print(f"   가중치 감쇠: {optimal_config.weight_decay} (정규화 강화)")
        print(f"   Dropout: {optimal_config.dropout} (과적합 방지)")
        print(f"   배치 크기: 64 (안정성)")
        print(f"   그래디언트 클리핑: 1.0")
        
        print(f"\n🔧 손실 함수 가중치:")
        print(f"   Entanglement: 1.0")
        print(f"   Fidelity: 10.0 (중요도 증가)")
        print(f"   Expressibility: 0.05 (큰 값이므로 가중치 감소)")
        
        print(f"\n⚙️  스케줄러 설정:")
        print(f"   ReduceLROnPlateau: patience=3, factor=0.5")
        print(f"   Early stopping: patience=10")
        
        return optimal_config
    
    def diagnose_plateau_causes(self):
        """1.5 손실 정체 원인 진단"""
        print(f"\n🔍 검증 손실 1.5 정체 원인 분석:")
        
        causes = [
            {
                "원인": "모델 과적합",
                "증상": "훈련 손실은 감소하지만 검증 손실 정체",
                "해결책": "모델 크기 축소, Dropout 증가, 정규화 강화"
            },
            {
                "원인": "학습률 부적절",
                "증상": "손실이 특정 값에서 진동",
                "해결책": "적응적 학습률 스케줄러 사용"
            },
            {
                "원인": "손실 함수 가중치 불균형",
                "증상": "일부 속성만 학습되고 다른 속성 무시",
                "해결책": "속성별 가중치 재조정"
            },
            {
                "원인": "데이터 품질 문제",
                "증상": "노이즈가 많거나 이상치 존재",
                "해결책": "데이터 전처리 및 이상치 제거"
            },
            {
                "원인": "그래디언트 소실/폭발",
                "증상": "그래디언트 노름이 너무 작거나 큼",
                "해결책": "그래디언트 클리핑 및 가중치 초기화 개선"
            }
        ]
        
        for i, cause in enumerate(causes, 1):
            print(f"   {i}. {cause['원인']}")
            print(f"      증상: {cause['증상']}")
            print(f"      해결책: {cause['해결책']}")
            print()
    
    def create_improved_model_config(self):
        """개선된 모델 설정 생성"""
        print(f"\n🚀 개선된 Property Prediction 설정:")
        
        # 핵심 개선사항 적용
        improved_config = PropertyPredictionConfig(
            # 모델 크기 최적화 (과적합 방지)
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024,
            
            # 정규화 강화
            dropout=0.4,
            weight_decay=1e-2,
            
            # 학습 최적화
            learning_rate=5e-4,
            warmup_steps=500,
            
            # 배치 설정
            train_batch_size=64,
            val_batch_size=64,
            
            # 출력 설정 (robust_fidelity 제거됨)
            property_dim=3  # entanglement, fidelity, expressibility
        )
        
        # 파라미터 수 확인
        test_model = PropertyPredictionTransformer(improved_config)
        total_params = sum(p.numel() for p in test_model.parameters())
        
        print(f"   최적화된 파라미터 수: {total_params:,}")
        print(f"   기존 대비 감소율: {(58_607_446 - total_params) / 58_607_446 * 100:.1f}%")
        
        return improved_config


def main():
    """메인 분석 실행"""
    print("🔍 Property Prediction 수렴 문제 종합 분석\n")
    
    analyzer = ConvergenceAnalyzer()
    
    # 1. 현재 설정 분석
    current_config = PropertyPredictionConfig()
    print("📋 현재 설정 분석:")
    analyzer.analyze_model_capacity(current_config)
    
    # 2. 정체 원인 진단
    analyzer.diagnose_plateau_causes()
    
    # 3. 최적 설정 권장
    optimal_config = analyzer.recommend_optimal_config()
    
    # 4. 개선된 설정 생성
    improved_config = analyzer.create_improved_model_config()
    
    print(f"\n✅ 분석 완료!")
    print(f"\n🎯 즉시 적용 가능한 해결책:")
    print(f"   1. 모델 크기 축소: d_model=256, n_layers=4")
    print(f"   2. 정규화 강화: dropout=0.4, weight_decay=1e-2")
    print(f"   3. 학습률 증가: 5e-4 (빠른 수렴)")
    print(f"   4. 손실 가중치 재조정: fidelity=10.0, expressibility=0.05")
    print(f"   5. 적응적 스케줄러: ReduceLROnPlateau")
    
    return improved_config


if __name__ == "__main__":
    improved_config = main()
