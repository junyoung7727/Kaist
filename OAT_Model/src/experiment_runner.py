"""
실험 실행기 - 다양한 모델 설정으로 자동 실험 실행
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.experiment_configs import get_experiment_config, list_experiments, create_experiment_configs
from config.unified_training_config import UnifiedTrainingConfig
from train_unified import main as train_main

def run_experiment(experiment_name: str, data_path: str = None, 
                  enable_rtg: bool = False, property_model_size: str = "medium", 
                  property_attention_mode: str = "standard", enable_augmentation: bool = True):
    from config.unified_training_config import UnifiedTrainingConfig
    import os
    """단일 실험 실행 (RTG 지원 포함)"""
    print(f"\n🚀 실험 시작: {experiment_name}")
    
    # 실험 설정 로드
    exp_config = get_experiment_config(experiment_name)
    
    # 기본 데이터 경로 설정
    if data_path is None:
        data_path = r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json"
    
    # 통합 설정 생성
    unified_config = UnifiedTrainingConfig()
    
    # 실험 설정으로 모델 파라미터 업데이트
    unified_config.model.d_model = exp_config.d_model
    unified_config.model.n_heads = exp_config.n_heads
    unified_config.model.n_layers = exp_config.n_layers
    unified_config.model.d_ff = exp_config.d_ff
    unified_config.model.dropout = exp_config.dropout
    unified_config.model.attention_mode = exp_config.attention_mode
    
    # 학습 파라미터 업데이트
    unified_config.training.learning_rate = exp_config.learning_rate
    unified_config.training.train_batch_size = exp_config.batch_size
    unified_config.training.val_batch_size = exp_config.batch_size
    unified_config.training.num_epochs = exp_config.num_epochs
    
    # RTG 및 데이터 증강 설정 추가
    unified_config.enable_rtg = enable_rtg
    unified_config.property_model_size = property_model_size
    unified_config.property_attention_mode = property_attention_mode
    unified_config.enable_augmentation = enable_augmentation
    
    # 저장 디렉토리 설정 (실험별로 구분)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/{exp_config.save_name}_{timestamp}"
    
    # 실험 정보 출력
    print(f"📋 실험 정보:")
    print(f"  - 모델 타입: {exp_config.model_type}")
    print(f"  - 어텐션 모드: {exp_config.attention_mode}")
    print(f"  - 모델 크기: {exp_config.model_size}")
    print(f"  - 파라미터: d_model={exp_config.d_model}, n_heads={exp_config.n_heads}, n_layers={exp_config.n_layers}")
    print(f"  - 배치 크기: {exp_config.batch_size}")
    print(f"  - 학습률: {exp_config.learning_rate}")
    print(f"  - 데이터 증강: {'활성화' if enable_augmentation else '비활성화'}")
    print(f"  - 저장 경로: {save_dir}")
    print()
    
    try:
        # 실험 설정 저장
        os.makedirs(save_dir, exist_ok=True)
        exp_info = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "config": {
                "model_type": exp_config.model_type,
                "attention_mode": exp_config.attention_mode,
                "model_size": exp_config.model_size,
                "d_model": exp_config.d_model,
                "n_heads": exp_config.n_heads,
                "n_layers": exp_config.n_layers,
                "d_ff": exp_config.d_ff,
                "dropout": exp_config.dropout,
                "learning_rate": exp_config.learning_rate,
                "batch_size": exp_config.batch_size,
                "num_epochs": exp_config.num_epochs
            }
        }
        
        with open(f"{save_dir}/experiment_info.json", 'w', encoding='utf-8') as f:
            json.dump(exp_info, f, indent=2, ensure_ascii=False)
        
        # 모델 타입에 따라 학습 실행
        if exp_config.model_type == "property":
            # Property 모델 학습
            from training.property_prediction_trainer import PropertyPredictionTrainer, create_datasets
            from models.property_prediction_transformer import create_property_prediction_model, PropertyPredictionConfig
            
            # Property 설정 생성
            prop_config = PropertyPredictionConfig(
                d_model=exp_config.d_model,
                n_heads=exp_config.n_heads,
                n_layers=exp_config.n_layers,
                d_ff=exp_config.d_ff,
                dropout=exp_config.dropout,
                attention_mode=exp_config.attention_mode,
                use_rotary_pe=True,
                max_qubits=10,
                train_batch_size=exp_config.batch_size,
                val_batch_size=exp_config.batch_size,
                learning_rate=exp_config.learning_rate,
                property_dim=3
            )
            
            # 모델 생성
            model = create_property_prediction_model(prop_config)
            
            # 데이터셋 로드 (증강 포함)
            train_dataset, val_dataset, test_dataset = create_datasets(data_path, enable_augmentation=True)
            
            # 트레이너 생성
            trainer = PropertyPredictionTrainer(
                model=model,
                config=prop_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                save_dir=save_dir
            )
            
            # 학습 실행
            results = trainer.train(num_epochs=exp_config.num_epochs)
            
        elif exp_config.model_type == "decision_transformer":
            # Decision Transformer 학습 (RTG 지원 포함)
            from training.trainer import DecisionTransformerTrainer, create_dt_datasets
            from models.decision_transformer import create_decision_transformer_model
            from config.unified_training_config import UnifiedTrainingConfig
            from preprocessing.rtg_calculator import RTGCalculator, create_rtg_calculator_from_checkpoint
            from config.experiment_configs import create_property_prediction_config, get_property_checkpoint_path
            import torch
            import os
            
            # RTG 모드 확인
            rtg_calculator = None
            if enable_rtg:
                print(f"🎯 RTG 모드 활성화")
                print(f"  - Property 모델 크기: {property_model_size}")
                print(f"  - 어텐션 모드: {property_attention_mode}")
                
                try:
                    # Property 모델 설정 생성
                    prop_config = create_property_prediction_config(
                        size=property_model_size,
                        attention_mode=property_attention_mode
                    )
                    
                    # Property 모델 체크포인트 경로
                    property_checkpoint_path = get_property_checkpoint_path(
                        size=property_model_size,
                        attention_mode=property_attention_mode
                    )
                    
                    if os.path.exists(property_checkpoint_path):
                        print(f"📥 Property 모델 로드: {property_checkpoint_path}")
                        
                        # RTG Calculator 생성
                        rtg_calculator = create_rtg_calculator_from_checkpoint(
                            checkpoint_path=property_checkpoint_path,
                            config=prop_config,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        
                        print(f"✅ RTG Calculator 생성 완료")
                    else:
                        print(f"⚠️ Property 모델 체크포인트를 찾을 수 없습니다: {property_checkpoint_path}")
                        print(f"⚠️ RTG 모드를 비활성화합니다.")
                        enable_rtg = False
                        
                except Exception as e:
                    print(f"⚠️ RTG Calculator 생성 오류: {e}")
                    print(f"⚠️ RTG 모드를 비활성화합니다.")
                    enable_rtg = False
                    rtg_calculator = None
            else:
                print(f"🔄 기본 모드 (비행동 복제)")
            
            # Decision Transformer 모델 설정
            dt_config = UnifiedTrainingConfig().model
            dt_config.d_model = exp_config.d_model
            dt_config.n_heads = exp_config.n_heads
            dt_config.n_layers = exp_config.n_layers
            
            # Decision Transformer 모델 생성 (RTG Calculator 전달)
            dt_model = create_decision_transformer_model(
                dt_config,
                rtg_calculator=rtg_calculator
            )
            
            # 데이터셋 로드 (증강 포함)
            train_dataset, val_dataset, test_dataset = create_dt_datasets(data_path, enable_augmentation=enable_augmentation)
            
            # DataLoader 생성 (RTG 지원)
            train_loader, val_loader, test_loader = create_dataloaders(
                train_dataset=train_dataset,
                val_dataset=val_dataset, 
                test_dataset=test_dataset,
                batch_size=exp_config.batch_size,
                num_workers=0,
                rtg_calculator=rtg_calculator,
                enable_rtg=enable_rtg
            )
            
            # 트레이너 생성
            trainer = DecisionTransformerTrainer(
                model=dt_model,
                config=dt_config,
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=save_dir,
                enable_rtg=enable_rtg
            )
            
            # 학습 실행
            results = trainer.train(num_epochs=exp_config.num_epochs)
        
        print(f"✅ 실험 완료: {experiment_name}")
        print(f"📊 최고 검증 손실: {results.get('best_val_loss', 'N/A')}")
        print(f"📁 결과 저장: {save_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 실험 실패: {experiment_name}")
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_experiments(data_path: str = None):
    """모든 실험 순차 실행"""
    experiments = list_experiments()
    
    print(f"🔬 총 {len(experiments)}개 실험 실행 시작")
    print("실험 목록:")
    for i, exp_name in enumerate(experiments, 1):
        print(f"  {i}. {exp_name}")
    print()
    
    results = {}
    for i, exp_name in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] 실험 실행 중...")
        success = run_experiment(exp_name, data_path)
        results[exp_name] = success
        
        if success:
            print(f"✅ {exp_name} 완료")
        else:
            print(f"❌ {exp_name} 실패")
    
    # 결과 요약
    print("\n" + "="*60)
    print("🏁 전체 실험 결과 요약")
    print("="*60)
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"성공: {successful}/{total}")
    
    for exp_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {exp_name}")

def run_property_size_comparison(data_path: str = None):
    """Property 모델 크기 비교 실험만 실행"""
    size_experiments = [
        "property_advanced_small",
        "property_advanced_medium", 
        "property_advanced_large"
    ]
    
    print("📊 Property 모델 크기 비교 실험")
    print("실험 목록:")
    for exp in size_experiments:
        print(f"  - {exp}")
    print()
    
    for exp_name in size_experiments:
        success = run_experiment(exp_name, data_path)
        if not success:
            print(f"⚠️ {exp_name} 실험 실패, 다음 실험 진행...")

def run_attention_comparison(data_path: str = None):
    """어텐션 모드 비교 실험만 실행"""
    attention_experiments = [
        "property_standard_medium",
        "property_advanced_medium",
        "decision_standard_medium", 
        "decision_advanced_medium"
    ]
    
    print("🔄 어텐션 모드 비교 실험")
    print("실험 목록:")
    for exp in attention_experiments:
        print(f"  - {exp}")
    print()
    
    for exp_name in attention_experiments:
        success = run_experiment(exp_name, data_path)
        if not success:
            print(f"⚠️ {exp_name} 실험 실패, 다음 실험 진행...")

def main():
    parser = argparse.ArgumentParser(description="모델 실험 실행기")
    parser.add_argument("--experiment", "-e", type=str, help="실행할 실험 이름")
    parser.add_argument("--all", action="store_true", help="모든 실험 실행")
    parser.add_argument("--size-comparison", action="store_true", help="Property 모델 크기 비교만 실행")
    parser.add_argument("--attention-comparison", action="store_true", help="어텐션 모드 비교만 실행")
    parser.add_argument("--list", action="store_true", help="사용 가능한 실험 목록 출력")
    parser.add_argument("--data-path", type=str, help="데이터 파일 경로")
    
    args = parser.parse_args()
    
    if args.list:
        experiments = list_experiments()
        print("🔬 사용 가능한 실험:")
        for exp in experiments:
            config = get_experiment_config(exp)
            print(f"  - {exp}: {config.model_type}, {config.attention_mode}, {config.model_size}")
        return
    
    if args.all:
        run_all_experiments(args.data_path)
    elif args.size_comparison:
        run_property_size_comparison(args.data_path)
    elif args.attention_comparison:
        run_attention_comparison(args.data_path)
    elif args.experiment:
        run_experiment(args.experiment, args.data_path)
    else:
        print("사용법:")
        print("  python experiment_runner.py --list                    # 실험 목록 보기")
        print("  python experiment_runner.py -e property_advanced_small # 단일 실험 실행")
        print("  python experiment_runner.py --size-comparison          # 크기 비교 실험")
        print("  python experiment_runner.py --attention-comparison     # 어텐션 비교 실험")
        print("  python experiment_runner.py --all                     # 모든 실험 실행")

if __name__ == "__main__":
    main()
