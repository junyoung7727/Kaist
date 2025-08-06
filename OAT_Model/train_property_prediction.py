#!/usr/bin/env python3
"""
Property Prediction Transformer 메인 학습 스크립트

CircuitSpec으로부터 얽힘도, fidelity, robust fidelity를 예측하는 
트랜스포머 모델의 학습을 실행합니다.

사용법:
    python train_property_prediction.py --data_path path/to/data.json --epochs 100
"""

import argparse
import torch
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.property_prediction_transformer import (
    PropertyPredictionTransformer,
    PropertyPredictionConfig,
    create_property_prediction_model
)
from src.training.property_prediction_trainer import (
    PropertyPredictionTrainer,
    PropertyPredictionDataset,
    create_datasets
)

def main():
    parser = argparse.ArgumentParser(description='Property Prediction Transformer 학습')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='C:\\Users\\jungh\\Documents\\GitHub\\Kaist\\results\\dummy_quantum_dataset.json',
                       help='학습 데이터 JSON 파일 경로')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256, help='모델 차원')
    parser.add_argument('--n_heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--n_layers', type=int, default=6, help='트랜스포머 레이어 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='가중치 감쇠')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='property_prediction_checkpoints', 
                       help='체크포인트 저장 디렉토리')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='학습 데이터 비율')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='검증 데이터 비율')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print("🧬 Property Prediction Transformer 학습 시작")
    print("=" * 60)
    print(f"📊 데이터 경로: {args.data_path}")
    print(f"🏗️  모델 설정: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"📚 학습 설정: epochs={args.epochs}, lr={args.learning_rate}, batch_size={args.batch_size}")
    print("=" * 60)
    
    # Check if data file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("다음 경로들을 확인해보세요:")
        possible_paths = [
            "../Ansatz_Data_ver2/results/unified_batch_experiment_results_with_circuits.json",
            "../results/unified_batch_experiment_results_with_circuits.json",
            "results/unified_batch_experiment_results_with_circuits.json"
        ]
        for path in possible_paths:
            if Path(path).exists():
                print(f"✅ 발견: {path}")
            else:
                print(f"❌ 없음: {path}")
        return
    
    try:
        # Create model configuration
        config = PropertyPredictionConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create model
        print("🏗️  모델 생성 중...")
        model = create_property_prediction_model(config)
        
        # Create datasets
        print("📊 데이터셋 생성 중...")
        train_dataset, val_dataset, test_dataset = create_datasets(
            str(data_path), 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio
        )
        
        print(f"✅ 데이터셋 분할 완료:")
        print(f"   - 학습: {len(train_dataset)} 샘플")
        print(f"   - 검증: {len(val_dataset)} 샘플") 
        print(f"   - 테스트: {len(test_dataset)} 샘플")
        
        # Create trainer
        print("🎯 트레이너 생성 중...")
        trainer = PropertyPredictionTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=args.save_dir
        )
        
        # Start training
        print("🚀 학습 시작!")
        trainer.train(num_epochs=args.epochs)
        
        print("🎉 학습 완료!")
        print(f"📁 결과 저장 위치: {args.save_dir}")
        
        # Test evaluation (optional)
        print("\n🧪 테스트 데이터 평가...")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=trainer.train_loader.collate_fn
        )
        
        # Load best model for testing
        best_model_path = Path(args.save_dir) / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=trainer.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 최고 모델 로드: {best_model_path}")
            
            # Simple test evaluation
            model.eval()
            test_losses = []
            
            with torch.no_grad():
                for batch in test_loader:
                    try:
                        circuit_specs = batch['circuit_specs']
                        targets = {k: v.to(trainer.device) for k, v in batch['targets'].items()}
                        
                        predictions = model(circuit_specs)
                        for key in predictions:
                            predictions[key] = predictions[key].to(trainer.device)
                        
                        losses = trainer.criterion(predictions, targets)
                        test_losses.append(losses['total'].item())
                        
                    except Exception as e:
                        print(f"⚠️ 테스트 배치 오류: {e}")
                        continue
            
            if test_losses:
                avg_test_loss = sum(test_losses) / len(test_losses)
                print(f"📊 테스트 손실: {avg_test_loss:.4f}")
            else:
                print("⚠️ 테스트 평가 실패")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
