"""
Decision Transformer Training Script
메인 실행 스크립트 - 간단하고 직관적인 사용법
"""

import argparse
import torch
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트 추가 (src 폴더를 최상위 패키지로 인식)
src_path = Path(__file__).parent
sys.path.append(str(src_path))

# 상위 디렉토리도 추가 (OAT_Model 폴더)
parent_path = src_path.parent
sys.path.append(str(parent_path))

# 임포트
from src.training.trainer import DecisionTransformerTrainer, TrainingConfig, set_seed, QuantumCircuitCollator
from src.data.quantum_circuit_dataset import DatasetManager, create_dataloaders
from src.data.embedding_pipeline import create_embedding_pipeline, EmbeddingConfig
from src.models.decision_transformer import create_decision_transformer


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer for Quantum Circuits")
    
    # 데이터 설정
    parser.add_argument("--path", type=str, 
                       default=r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\raw_data\merged_data.json",
                       help="Path to unified data JSON file (contains both merged_results and merged_circuits)")
    # 레거시 지원을 위한 개별 파일 경로
    parser.add_argument("--merged_results_path", type=str, 
                       default=None,
                       help="Path to merged results JSON file (legacy support)")
    parser.add_argument("--circuits_path", type=str, 
                       default=None,
                       help="Path to circuit specifications JSON file (legacy support)")
    parser.add_argument("--enable_filtering", action="store_true", default=True,
                       help="Enable data quality filtering (remove invalid expressibility data)")
    
    # 모델 설정
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_gate_types", type=int, default=20, help="Number of gate types")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # 학습 설정 (최적화됨)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate (최적화: 1e-4 → 3e-4)")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs (최적화: 1 → 20)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # 기타 설정
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint save directory")
    
    # GPU 메모리 최적화 설정
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--memory_cleanup_interval", type=int, default=50, help="Memory cleanup interval (batches)")
    
    # 로깅 설정
    parser.add_argument("--use_wandb", action="store_true",default=True, help="Use Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="quantum-decision-transformer", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    
    # 데이터셋 분할 설정
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test data ratio")
    
    args = parser.parse_args()
    
    # WANDB_API_KEY 환경변수 검사
    if args.use_wandb and not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY environment variable not set. Disabling wandb logging.")
        args.use_wandb = False
    
    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("🚀 Quantum Decision Transformer Training")
    print("=" * 60)
    print(f"Device: {device}")
    if args.path:
        print(f"Unified data path: {args.path}")
    print(f"Data filtering: {args.enable_filtering}")
    print(f"Model: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"Training: batch_size={args.batch_size}, epochs={args.num_epochs}, lr={args.learning_rate}")
    print("=" * 60)
    
    # 시드 설정
    set_seed(args.seed)
    
    # 1. 데이터셋 준비
    print("📊 Loading and preparing dataset...")
    if args.path:
        manager = DatasetManager(
            unified_data_path=args.path
        )
    
    # 데이터 병합 및 품질 필터링 (에러 발생 시 즉시 중단)
    circuit_data = manager.merge_data(enable_filtering=args.enable_filtering)
    stats = manager.get_dataset_stats()
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total valid circuits: {stats['total_circuits']}")
    print(f"   Qubit range: {stats['qubit_range']}")
    print(f"   Gate range: {stats['gate_range']}")
    print(f"   Fidelity range: {stats['fidelity_range']}")
    if 'entanglement_range' in stats:
        print(f"   Entanglement range: {stats['entanglement_range']}")
    train_dataset, val_dataset, test_dataset = manager.split_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} circuits")
    print(f"   Validation: {len(val_dataset)} circuits")
    print(f"   Test: {len(test_dataset)} circuits")
    
    # 첫 번째 샘플 확인
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\n📋 Sample circuit info:")
        print(f"   Circuit ID: {sample.circuit_id}")
        print(f"   Qubits: {sample.num_qubits}, Gates: {len(sample.gates)}")
        print(f"   Fidelity: {sample.measurement_result.fidelity:.4f}")
        print(f"   Entanglement: {sample.measurement_result.entanglement:.4f}")
        if sample.measurement_result.expressibility:
            expr = sample.measurement_result.expressibility
            print(f"   Expressibility: {expr.get('expressibility', 'N/A'):.4f}")
            print(f"   KL Divergence: {expr.get('kl_divergence', 'N/A'):.4f}")
    
    # 2. 임베딩 파이프라인 설정
    print("\n🔧 Setting up embedding pipeline...")
    embed_config = EmbeddingConfig(
        d_model=args.d_model,
        n_gate_types=args.n_gate_types,
        max_seq_len=2000
    )
    

    embedding_pipeline = create_embedding_pipeline(embed_config)
    print("✅ Embedding pipeline created successfully!")


    # 3. 데이터로더 생성 (에러 발생 시 즉시 중단)
    print("\n📦 Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=0  # 🚀 FIX: RLock pickle 오류 방지 (캐싱 시스템과 멀티프로세싱 충돌)
    )
    
    # 콜레이터 설정
    collator = QuantumCircuitCollator(embedding_pipeline)
    train_loader.collate_fn = collator
    val_loader.collate_fn = collator
    
    print("✅ Data loaders created successfully!")
    
    # 4. 모델 생성 (에러 발생 시 즉시 중단)
    print("\n🤖 Creating Decision Transformer model...")
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
        run_name=args.run_name,
        # 🚀 AMP 설정 명시적 추가
        use_amp=False,  # Mixed Precision 활성화
        gradient_accumulation_steps=1,
        gradient_checkpointing=True
    )
    
    # 6. 트레이너 생성 및 학습 시작 (에러 발생 시 즉시 중단)
    print("\n🎯 Starting training...")
    try:
        trainer = DecisionTransformerTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=config,
            embedding_pipeline=embedding_pipeline
        )
        
        # 학습 시작
        trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"💾 Checkpoints saved in: {args.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")


if __name__ == "__main__":
    main()
