"""
빠른 실험 실행 스크립트 - 간단한 명령어로 다양한 모델 설정 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.unified_training_config import UnifiedConfig
from datetime import datetime

def create_experiment_config(model_size: str, attention_mode: str = "advanced"):
    """실험용 설정 생성"""
    
    # 6800개 데이터에 맞는 모델 크기 설정
    size_configs = {
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
    
    if model_size not in size_configs:
        raise ValueError(f"지원하지 않는 모델 크기: {model_size}. 사용 가능: {list(size_configs.keys())}")
    
    size_config = size_configs[model_size]
    
    # 통합 설정 생성
    config = UnifiedConfig()
    
    # 모델 설정 업데이트
    config.model.d_model = size_config["d_model"]
    config.model.n_heads = size_config["n_heads"]
    config.model.n_layers = size_config["n_layers"]
    config.model.d_ff = size_config["d_ff"]
    config.model.dropout = size_config["dropout"]
    config.model.attention_mode = attention_mode
    
    # 학습 설정 업데이트
    config.training.train_batch_size = size_config["batch_size"]
    config.training.val_batch_size = size_config["batch_size"]
    config.training.num_epochs = 100
    config.training.learning_rate = 1e-4
    
    return config

def run_property_experiment(model_size: str, attention_mode: str = "advanced"):
    """Property 모델 실험 실행"""
    print(f"\n🚀 Property 모델 실험 시작")
    print(f"📊 모델 크기: {model_size}")
    print(f"🔄 어텐션 모드: {attention_mode}")
    print("=" * 50)
    
    # 설정 생성
    config = create_experiment_config(model_size, attention_mode)
    
    # 실험 정보 출력
    print(f"🔧 모델 설정:")
    print(f"  - d_model: {config.model.d_model}")
    print(f"  - n_heads: {config.model.n_heads}")
    print(f"  - n_layers: {config.model.n_layers}")
    print(f"  - dropout: {config.model.dropout}")
    print(f"  - batch_size: {config.training.train_batch_size}")
    print()
    
    try:
        # Property 모델 학습
        from training.property_prediction_trainer import PropertyPredictionTrainer, create_datasets
        from models.property_prediction_transformer import create_property_prediction_model, PropertyPredictionConfig
        
        # Property 설정 생성
        prop_config = PropertyPredictionConfig(
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            attention_mode=attention_mode,
            use_rotary_pe=True,
            max_qubits=10,
            train_batch_size=config.training.train_batch_size,
            val_batch_size=config.training.val_batch_size,
            learning_rate=config.training.learning_rate,
            property_dim=3
        )
        
        # 모델 생성
        model = create_property_prediction_model(prop_config)
        
        # 데이터셋 로드
        data_path = r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json"
        train_dataset, val_dataset, test_dataset = create_datasets(data_path)
        
        # 저장 디렉토리 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"property_{attention_mode}_{model_size}_{config.model.d_model}d_{config.model.n_layers}l_{timestamp}"
        save_dir = f"experiments/{save_name}"
        
        print(f"💾 저장 경로: {save_dir}")
        
        # 트레이너 생성
        trainer = PropertyPredictionTrainer(
            model=model,
            config=prop_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=save_dir
        )
        
        # 학습 실행
        results = trainer.train(num_epochs=config.training.num_epochs)
        
        print(f"\n✅ 실험 완료!")
        print(f"📊 최고 검증 손실: {results.get('best_val_loss', 'N/A'):.4f}")
        print(f"📁 결과 저장: {save_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_size_comparison():
    """모델 크기별 성능 비교"""
    print("\n🔬 Property 모델 크기 비교 실험")
    print("=" * 60)
    
    sizes = ["small", "medium", "large"]
    results = {}
    
    for size in sizes:
        print(f"\n[{sizes.index(size)+1}/{len(sizes)}] {size.upper()} 모델 실험 중...")
        success = run_property_experiment(size, "advanced")
        results[size] = success
        
        if success:
            print(f"✅ {size} 모델 완료")
        else:
            print(f"❌ {size} 모델 실패")
    
    # 결과 요약
    print("\n" + "="*60)
    print("🏁 크기 비교 실험 결과")
    print("="*60)
    
    for size, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {size.upper()} 모델")

def run_attention_comparison():
    """어텐션 모드 비교 (Medium 크기)"""
    print("\n🔄 어텐션 모드 비교 실험 (Medium 크기)")
    print("=" * 60)
    
    attention_modes = ["basic", "advanced"]
    results = {}
    
    for mode in attention_modes:
        print(f"\n[{attention_modes.index(mode)+1}/{len(attention_modes)}] {mode.upper()} 어텐션 실험 중...")
        success = run_property_experiment("medium", mode)
        results[mode] = success
        
        if success:
            print(f"✅ {mode} 어텐션 완료")
        else:
            print(f"❌ {mode} 어텐션 실패")
    
    # 결과 요약
    print("\n" + "="*60)
    print("🏁 어텐션 비교 실험 결과")
    print("="*60)
    
    for mode, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {mode.upper()} 어텐션")

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("🔬 빠른 실험 실행기")
        print("\n사용법:")
        print("  python quick_experiment.py single <size> [attention_mode]")
        print("    - 단일 실험: python quick_experiment.py single medium")
        print("    - 어텐션 지정: python quick_experiment.py single medium basic")
        print()
        print("  python quick_experiment.py size-comparison")
        print("    - 모델 크기 비교 (small, medium, large)")
        print()
        print("  python quick_experiment.py attention-comparison") 
        print("    - 어텐션 모드 비교 (basic vs advanced)")
        print()
        print("📊 사용 가능한 모델 크기: small, medium, large")
        print("🔄 사용 가능한 어텐션 모드: basic, advanced")
        return
    
    command = sys.argv[1]
    
    if command == "single":
        if len(sys.argv) < 3:
            print("❌ 모델 크기를 지정해주세요: small, medium, large")
            return
        
        model_size = sys.argv[2]
        attention_mode = sys.argv[3] if len(sys.argv) > 3 else "advanced"
        
        run_property_experiment(model_size, attention_mode)
        
    elif command == "size-comparison":
        run_size_comparison()
        
    elif command == "attention-comparison":
        run_attention_comparison()
        
    else:
        print(f"❌ 알 수 없는 명령어: {command}")
        print("사용 가능한 명령어: single, size-comparison, attention-comparison")

if __name__ == "__main__":
    main()
