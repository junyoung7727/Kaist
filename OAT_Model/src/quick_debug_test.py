"""
Quick Debug Test for Property Prediction Training
핵심 문제 빠른 진단
"""

import torch
import sys
from pathlib import Path
import json

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))

from debug_train_property import PropertyTrainingAnalyzer
from models.property_prediction_transformer import PropertyPredictionConfig


def quick_convergence_analysis():
    """빠른 수렴 문제 분석"""
    print("🔍 Property Prediction 수렴 문제 빠른 진단\n")
    
    # 1. 모델 설정 검토
    config = PropertyPredictionConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        dropout=0.3,
        learning_rate=1e-4,
        weight_decay=1e-3
    )
    
    print("📋 현재 모델 설정:")
    print(f"   d_model: {config.d_model}")
    print(f"   n_layers: {config.n_layers}")
    print(f"   dropout: {config.dropout}")
    print(f"   learning_rate: {config.learning_rate}")
    print(f"   weight_decay: {config.weight_decay}")
    
    # 2. 모델 용량 분석
    analyzer = PropertyTrainingAnalyzer(model_config=config, debug_mode="minimal")
    total_params = sum(p.numel() for p in analyzer.model.parameters())
    trainable_params = sum(p.numel() for p in analyzer.model.parameters() if p.requires_grad)
    
    print(f"\n🧠 모델 용량:")
    print(f"   총 파라미터: {total_params:,}")
    print(f"   학습 가능: {trainable_params:,}")
    
    # 3. 가능한 문제점 분석
    print(f"\n🔍 잠재적 문제점 분석:")
    
    # 학습률 문제
    if config.learning_rate > 1e-3:
        print("   ⚠️  학습률이 높을 수 있음 (>1e-3)")
    elif config.learning_rate < 1e-5:
        print("   ⚠️  학습률이 낮을 수 있음 (<1e-5)")
    else:
        print("   ✅ 학습률 적절")
    
    # 모델 크기 문제
    if total_params > 50_000_000:  # 50M
        print("   ⚠️  모델이 클 수 있음 (과적합 위험)")
    elif total_params < 1_000_000:  # 1M
        print("   ⚠️  모델이 작을 수 있음 (용량 부족)")
    else:
        print("   ✅ 모델 크기 적절")
    
    # Dropout 문제
    if config.dropout > 0.5:
        print("   ⚠️  Dropout이 높을 수 있음 (학습 방해)")
    elif config.dropout < 0.1:
        print("   ⚠️  Dropout이 낮을 수 있음 (과적합 위험)")
    else:
        print("   ✅ Dropout 적절")
    
    # 4. 권장 설정
    print(f"\n💡 수렴 개선 권장 설정:")
    print("   1. 학습률 스케줄러: ReduceLROnPlateau (patience=5)")
    print("   2. 그래디언트 클리핑: max_norm=1.0")
    print("   3. 가중치 초기화: Xavier uniform (gain=0.1)")
    print("   4. 손실 함수 가중치 재조정:")
    print("      - Fidelity: 5.0 (중요도 증가)")
    print("      - Expressibility: 0.1 (가중치 감소)")
    print("   5. 배치 크기: 32-64 (안정성)")
    print("   6. Early stopping: patience=15")
    
    # 5. 데이터 관련 체크포인트
    print(f"\n📊 데이터 품질 체크포인트:")
    print("   1. 타겟 값 범위 확인:")
    print("      - Entanglement: [0, 1]")
    print("      - Fidelity: [0, 1]") 
    print("      - Expressibility: [0, ~50]")
    print("   2. 이상치 제거 필요성")
    print("   3. 데이터 정규화 상태")
    print("   4. 배치별 타겟 분포 균형")
    
    return analyzer


def test_with_dummy_data():
    """더미 데이터로 빠른 테스트"""
    print("\n🧪 더미 데이터 테스트...")
    
    analyzer = quick_convergence_analysis()
    
    # 더미 배치 생성
    batch_size = 4
    device = analyzer.device
    
    # Circuit 객체 클래스 정의 (호환성을 위해)
    class DummyCircuit:
        def __init__(self, nodes, n_qubits, depth):
            self.nodes = nodes
            self.num_qubits = n_qubits  # grid_encoder가 기대하는 속성명
            self.n_qubits = n_qubits
            self.depth = depth
    
    # 더미 circuit specs (Circuit 객체로 생성)
    dummy_circuit_specs = []
    for i in range(batch_size):
        nodes = [
            {'gate_name': 'H', 'qubits': [0], 'parameter_value': 0.0},
            {'gate_name': 'RX', 'qubits': [0], 'parameter_value': 1.57},
            {'gate_name': 'CX', 'qubits': [0, 1], 'parameter_value': 0.0}
        ]
        circuit = DummyCircuit(nodes=nodes, n_qubits=2, depth=3)
        dummy_circuit_specs.append(circuit)
    
    # 더미 타겟
    dummy_targets = {
        'entanglement': torch.rand(batch_size, device=device) * 0.8 + 0.1,  # [0.1, 0.9]
        'fidelity': torch.rand(batch_size, device=device) * 0.6 + 0.3,      # [0.3, 0.9]
        'expressibility': torch.rand(batch_size, device=device) * 20 + 5     # [5, 25]
    }
    
    print(f"   배치 크기: {batch_size}")
    print(f"   디바이스: {device}")
    
    try:
        # Forward pass 테스트
        analyzer.model.eval()
        with torch.no_grad():
            predictions = analyzer.model(dummy_circuit_specs)
        
        print("   ✅ Forward pass 성공")
        
        # 예측 형태 확인
        for prop, pred in predictions.items():
            if prop != 'combined':
                target = dummy_targets.get(prop)
                if target is not None:
                    print(f"   {prop}: pred_shape={pred.shape}, target_shape={target.shape}")
        
        # 손실 계산 테스트
        loss_dict = analyzer.criterion(predictions, dummy_targets)
        total_loss = loss_dict['total']
        
        print(f"   ✅ 손실 계산 성공: {total_loss.item():.6f}")
        
        # 개별 손실 확인
        for prop, loss_val in loss_dict.items():
            if prop != 'total':
                print(f"      {prop}: {loss_val.item():.6f}")
        
        # Backward pass 테스트
        analyzer.model.train()
        analyzer.optimizer.zero_grad()
        total_loss.backward()
        
        # 그래디언트 체크
        grad_norm = torch.nn.utils.clip_grad_norm_(analyzer.model.parameters(), max_norm=1.0)
        print(f"   ✅ Backward pass 성공, grad_norm: {grad_norm:.6f}")
        
        analyzer.optimizer.step()
        print("   ✅ 옵티마이저 스텝 성공")
        
    except Exception as e:
        print(f"   ❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 빠른 진단 실행
    test_with_dummy_data()
    
    print(f"\n🎯 다음 단계:")
    print("   1. 실제 데이터로 debug_train_property.py 실행")
    print("   2. 손실 가중치 및 학습률 조정")
    print("   3. 데이터 품질 분석 결과 확인")
    print("   4. 그래디언트 및 수렴 패턴 모니터링")
