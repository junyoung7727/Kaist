"""
Decision Transformer Training Script
메인 실행 스크립트 - 간단하고 직관적인 사용법
"""

import argparse
import torch
from pathlib import Path
import sys

# 프로젝트 루트 추가 (src 폴더를 최상위 패키지로 인식)
src_path = Path(__file__).parent
sys.path.append(str(src_path))

# 상위 디렉토리도 추가 (OAT_Model 폴더)
parent_path = src_path.parent
sys.path.append(str(parent_path))

# 임포트 경로 문제 해결
try:
    # 절대 경로 시도
    from training.trainer import DecisionTransformerTrainer, TrainingConfig, set_seed, QuantumCircuitCollator
    from data.quantum_circuit_dataset import DatasetManager, create_dataloaders
    from data.embedding_pipeline import create_embedding_pipeline, EmbeddingConfig
    from models.decision_transformer import create_decision_transformer
except ImportError:
    # 상대 경로 시도
    from src.training.trainer import DecisionTransformerTrainer, TrainingConfig, set_seed, QuantumCircuitCollator
    from src.data.quantum_circuit_dataset import DatasetManager, create_dataloaders
    from src.data.embedding_pipeline import create_embedding_pipeline, EmbeddingConfig
    from src.models.decision_transformer import create_decision_transformer


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer for Quantum Circuits")
    
    # 데이터 설정
    parser.add_argument("--data_path", type=str, 
                       default="C:\\Users\\jungh\\Documents\\GitHub\\Kaist\\results\\dummy_quantum_dataset.json",
                       help="Path to quantum circuit data JSON file")
    
    # 모델 설정
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_gate_types", type=int, default=20, help="Number of gate types")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # 학습 설정
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # 기타 설정
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint save directory")
    
    # 로깅 설정
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="quantum-decision-transformer", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    
    # 데이터셋 분할 설정
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test data ratio")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("🚀 Quantum Decision Transformer Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Model: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"Training: batch_size={args.batch_size}, epochs={args.num_epochs}, lr={args.learning_rate}")
    print("=" * 60)
    
    # 시드 설정
    set_seed(args.seed)
    
    # 1. 데이터셋 준비
    print("📊 Loading and preparing dataset...")
    manager = DatasetManager(args.data_path)
    
    try:
        # 데이터 로딩 및 정보 출력
        circuit_specs = manager.parse_circuits()
        info = manager.get_dataset_info()
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total circuits: {info['total_circuits']}")
        print(f"   Qubits range: {info['num_qubits_range']}")
        print(f"   Gates range: {info['gate_count_range']}")
        print(f"   Gate types: {info['unique_gate_types']}")
        
        # 데이터셋 분할
        train_ds, val_ds, test_ds = manager.split_dataset(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        print(f"📊 Dataset split:")
        print(f"   Train: {len(train_ds)} circuits")
        print(f"   Validation: {len(val_ds)} circuits")
        print(f"   Test: {len(test_ds)} circuits")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 2. 임베딩 파이프라인 설정
    print("\n🔧 Setting up embedding pipeline...")
    embed_config = EmbeddingConfig(
        d_model=args.d_model,
        n_gate_types=args.n_gate_types,
        max_seq_len=1000
    )
    

    embedding_pipeline = create_embedding_pipeline(embed_config)
    print("✅ Embedding pipeline created successfully!")


    # 3. 데이터로더 생성
    print("\n📦 Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=args.batch_size,
            num_workers=0
        )
        
        # 콜레이터 설정
        collator = QuantumCircuitCollator(embedding_pipeline)
        train_loader.collate_fn = collator
        val_loader.collate_fn = collator
        
        print("✅ Data loaders created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return
    
    # 4. 모델 생성
    print("\n🤖 Creating Decision Transformer model...")
    try:
        model = create_decision_transformer(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_gate_types=args.n_gate_types,
            dropout=args.dropout
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✅ Model created successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # 5. 학습 설정
    print("\n⚙️ Setting up training configuration...")
    config = TrainingConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_gate_types=args.n_gate_types,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=device,
        seed=args.seed,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        run_name=args.run_name
    )
    
    # 6. 트레이너 생성 및 학습 시작
    print("\n🎯 Starting training...")
    try:
        trainer = DecisionTransformerTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=args.save_dir
        )
        
        # 학습 시작
        trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"💾 Checkpoints saved in: {args.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
