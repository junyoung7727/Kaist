"""
Hyperparameter Grid Search for Decision Transformer
하이퍼파라미터 그리드 서치 및 어텐션 메커니즘 비교 시스템
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import argparse
from typing import Dict, Any, List, Tuple
import time
import numpy as np
from dataclasses import asdict, dataclass
import itertools
from datetime import datetime
import os
# import pandas as pd  # 선택적 의존성

# 프로젝트 모듈 임포트
from src.models.decision_transformer import DecisionTransformer
from src.training.trainer import TrainingConfig, DecisionTransformerTrainer, QuantumCircuitCollator
from src.data.embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
from src.data.quantum_circuit_dataset import DatasetManager, create_dataloaders
from utils.debug_utils import debug_print

# 🎆 NEW: 게이트 레지스트리 싱글톤 임포트
sys.path.append(str(Path(__file__).parent / "quantumcommon"))
from gates import QuantumGateRegistry


@dataclass
class GridSearchConfig:
    """그리드 서치 설정 (핵심 파라미터만)"""
    # 핵심 하이퍼파라미터 범위
    n_layers_options: List[int] = None
    attention_mode_options: List[str] = None
    
    # 고정 하이퍼파라미터 (사용자 피드백에 따라 고정)
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1
    learning_rate: float = 1e-4
    
    # 훈련 설정
    num_epochs: int = 10
    batch_size: int = 16
    max_experiments: int = 50  # 최대 실험 수 제한
    
    # 데이터 및 저장 경로
    data_path: str = "data/dummy_experiment_results.json"
    results_dir: str = "grid_search_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """기본값 설정"""
        if self.n_layers_options is None:
            self.n_layers_options = [4, 6, 8, 10]  # 레이어 수가 가장 중요
        if self.attention_mode_options is None:
            self.attention_mode_options = ["standard", "advanced", "hybrid"]


class HyperparameterGridSearch:
    """하이퍼파라미터 그리드 서치 클래스"""
    
    def __init__(self, config: GridSearchConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 결과 저장
        self.experiment_results = []
        self.best_config = None
        self.best_val_loss = float('inf')
        
        print(f"🔬 Grid Search initialized")
        print(f"   📁 Results directory: {self.results_dir}")
        print(f"   🎯 Device: {self.device}")
    
    def generate_hyperparameter_combinations(self) -> List[Dict[str, Any]]:
        """하이퍼파라미터 조합 생성 (핵심 파라미터만)"""
        param_names = ['n_layers', 'attention_mode']
        param_values = [
            self.config.n_layers_options,
            self.config.attention_mode_options
        ]
        
        # 모든 조합 생성
        combinations = list(itertools.product(*param_values))
        
        # 조합을 딕셔너리로 변환
        valid_combinations = []
        for combo in combinations:
            n_layers, attention_mode = combo
            config_dict = {
                'n_layers': n_layers,
                'attention_mode': attention_mode,
                # 고정 파라미터 추가
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'dropout': self.config.dropout,
                'learning_rate': self.config.learning_rate
            }
            valid_combinations.append(config_dict)
        
        # 최대 실험 수 제한
        if len(valid_combinations) > self.config.max_experiments:
            print(f"⚠️ Too many combinations ({len(valid_combinations)}), limiting to {self.config.max_experiments}")
            # 랜덤 샘플링으로 제한
            import random
            random.shuffle(valid_combinations)
            valid_combinations = valid_combinations[:self.config.max_experiments]
        
        print(f"📊 Generated {len(valid_combinations)} valid hyperparameter combinations")
        return valid_combinations
    
    def create_training_config(self, hyperparams: Dict[str, Any]) -> TrainingConfig:
        """하이퍼파라미터로부터 TrainingConfig 생성"""
        return TrainingConfig(
            d_model=hyperparams['d_model'],  # 고정값
            n_layers=hyperparams['n_layers'],  # 변수
            n_heads=hyperparams['n_heads'],  # 고정값
            dropout=hyperparams['dropout'],  # 고정값
            learning_rate=hyperparams['learning_rate'],  # 고정값
            attention_mode=hyperparams['attention_mode'],  # 변수
            
            # 고정 설정
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            device=self.config.device,
            use_wandb=False,  # 그리드 서치에서는 wandb 비활성화
            # n_gate_types는 TrainingConfig.__post_init__에서 자동 설정됨
            
            # 빠른 실험을 위한 설정
            eval_every=100,
            save_every=1000
        )
    
    def run_single_experiment(self, experiment_id: int, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """단일 실험 실행"""
        print(f"\n🧪 Experiment {experiment_id + 1}")
        print(f"   🔧 Config: {hyperparams}")
        
        start_time = time.time()
        
        try:
            # 훈련 설정 생성
            train_config = self.create_training_config(hyperparams)
            
            # 임베딩 파이프라인 설정
            embedding_config = EmbeddingConfig(
                d_model=train_config.d_model,
                n_gate_types=train_config.n_gate_types
            )
            embedding_pipeline = EmbeddingPipeline(embedding_config)
            
            # 데이터 로더 생성
            dataset_manager = DatasetManager(self.config.data_path)
            train_loader, val_loader = create_dataloaders(
                dataset_manager=dataset_manager,
                embedding_pipeline=embedding_pipeline,
                batch_size=train_config.batch_size,
                train_split=0.8
            )
            
            # 모델 생성
            model = DecisionTransformer(
                d_model=train_config.d_model,
                n_layers=train_config.n_layers,
                n_heads=train_config.n_heads,
                n_gate_types=train_config.n_gate_types,
                dropout=train_config.dropout,
                attention_mode=train_config.attention_mode
            )
            
            # 트레이너 생성
            trainer = DecisionTransformerTrainer(
                config=train_config,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=str(self.results_dir / f"experiment_{experiment_id}")
            )
            
            # 훈련 실행
            trainer.train()
            
            # 최종 검증 성능 측정
            final_val_metrics = trainer.validate()
            
            experiment_time = time.time() - start_time
            
            # 결과 정리
            result = {
                'experiment_id': experiment_id,
                'hyperparameters': hyperparams,
                'final_val_loss': final_val_metrics['val_loss'],
                'final_val_accuracy': final_val_metrics['val_accuracy'],
                'best_val_loss': trainer.best_val_loss,
                'total_steps': trainer.global_step,
                'experiment_time': experiment_time,
                'success': True,
                'error': None
            }
            
            print(f"   ✅ Success! Val Loss: {final_val_metrics['val_loss']:.4f}, Accuracy: {final_val_metrics['val_accuracy']:.4f}")
            
            # 최고 성능 업데이트
            if final_val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = final_val_metrics['val_loss']
                self.best_config = hyperparams.copy()
                print(f"   🏆 New best configuration! Val Loss: {self.best_val_loss:.4f}")
            
        except Exception as e:
            experiment_time = time.time() - start_time
            print(f"   ❌ Failed: {str(e)}")
            
            result = {
                'experiment_id': experiment_id,
                'hyperparameters': hyperparams,
                'final_val_loss': float('inf'),
                'final_val_accuracy': 0.0,
                'best_val_loss': float('inf'),
                'total_steps': 0,
                'experiment_time': experiment_time,
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def run_grid_search(self) -> Dict[str, Any]:
        """전체 그리드 서치 실행"""
        print(f"🚀 Starting Hyperparameter Grid Search")
        print(f"   📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 하이퍼파라미터 조합 생성
        combinations = self.generate_hyperparameter_combinations()
        
        total_start_time = time.time()
        
        # 각 조합에 대해 실험 실행
        for i, hyperparams in enumerate(combinations):
            result = self.run_single_experiment(i, hyperparams)
            self.experiment_results.append(result)
            
            # 중간 결과 저장
            if (i + 1) % 5 == 0:
                self.save_intermediate_results()
        
        total_time = time.time() - total_start_time
        
        # 최종 결과 정리
        final_results = {
            'config': asdict(self.config),
            'total_experiments': len(combinations),
            'successful_experiments': sum(1 for r in self.experiment_results if r['success']),
            'failed_experiments': sum(1 for r in self.experiment_results if not r['success']),
            'total_time': total_time,
            'best_config': self.best_config,
            'best_val_loss': self.best_val_loss,
            'all_results': self.experiment_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # 결과 저장
        self.save_final_results(final_results)
        
        # 요약 출력
        self.print_summary(final_results)
        
        return final_results
    
    def save_intermediate_results(self):
        """중간 결과 저장"""
        intermediate_path = self.results_dir / "intermediate_results.json"
        with open(intermediate_path, 'w') as f:
            json.dump(self.experiment_results, f, indent=2)
        print(f"💾 Intermediate results saved to {intermediate_path}")
    
    def save_final_results(self, results: Dict[str, Any]):
        """최종 결과 저장"""
        # JSON 저장
        json_path = self.results_dir / "final_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # CSV 저장 (분석용)
        csv_path = self.results_dir / "results_summary.csv"
        self.save_results_as_csv(csv_path)
        
        print(f"💾 Final results saved to:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📊 CSV: {csv_path}")
    
    def save_results_as_csv(self, csv_path: Path):
        """결과를 CSV로 저장"""
        try:
            import pandas as pd
            rows = []
            for result in self.experiment_results:
                if result['success']:
                    row = {
                        'experiment_id': result['experiment_id'],
                        'final_val_loss': result['final_val_loss'],
                        'final_val_accuracy': result['final_val_accuracy'],
                        'best_val_loss': result['best_val_loss'],
                        'experiment_time': result['experiment_time'],
                        **result['hyperparameters']
                    }
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
        except ImportError:
            print("⚠️ pandas not available, skipping CSV export. Install with: pip install pandas")
            # 대신 간단한 CSV 수동 생성
            self._save_simple_csv(csv_path)
    
    def _save_simple_csv(self, csv_path: Path):
        """
pandas 없이 간단한 CSV 저장"""
        successful_results = [r for r in self.experiment_results if r['success']]
        if not successful_results:
            return
        
        # 헤더 생성
        headers = ['experiment_id', 'final_val_loss', 'final_val_accuracy', 'best_val_loss', 'experiment_time']
        param_keys = list(successful_results[0]['hyperparameters'].keys())
        headers.extend(param_keys)
        
        with open(csv_path, 'w') as f:
            # 헤더 쓰기
            f.write(','.join(headers) + '\n')
            
            # 데이터 쓰기
            for result in successful_results:
                row = [
                    str(result['experiment_id']),
                    str(result['final_val_loss']),
                    str(result['final_val_accuracy']),
                    str(result['best_val_loss']),
                    str(result['experiment_time'])
                ]
                for key in param_keys:
                    row.append(str(result['hyperparameters'][key]))
                f.write(','.join(row) + '\n')
    
    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("📈 GRID SEARCH SUMMARY")
        print("="*60)
        
        print(f"🔬 Total Experiments: {results['total_experiments']}")
        print(f"✅ Successful: {results['successful_experiments']}")
        print(f"❌ Failed: {results['failed_experiments']}")
        print(f"⏱️ Total Time: {results['total_time']/3600:.2f} hours")
        
        if results['best_config']:
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   📊 Validation Loss: {results['best_val_loss']:.4f}")
            for key, value in results['best_config'].items():
                print(f"   🔧 {key}: {value}")
        
        # 상위 5개 결과
        successful_results = [r for r in self.experiment_results if r['success']]
        if successful_results:
            top_5 = sorted(successful_results, key=lambda x: x['final_val_loss'])[:5]
            print(f"\n🥇 TOP 5 CONFIGURATIONS:")
            for i, result in enumerate(top_5, 1):
                print(f"   {i}. Loss: {result['final_val_loss']:.4f} | "
                      f"Attention: {result['hyperparameters']['attention_mode']} | "
                      f"d_model: {result['hyperparameters']['d_model']} | "
                      f"n_layers: {result['hyperparameters']['n_layers']}")
        
        # 어텐션 모드별 성능 비교
        self.analyze_attention_performance()
    
    def analyze_attention_performance(self):
        """어텐션 모드별 성능 분석"""
        successful_results = [r for r in self.experiment_results if r['success']]
        if not successful_results:
            return
        
        attention_stats = {}
        for result in successful_results:
            mode = result['hyperparameters']['attention_mode']
            if mode not in attention_stats:
                attention_stats[mode] = []
            attention_stats[mode].append(result['final_val_loss'])
        
        print(f"\n🎯 ATTENTION MODE PERFORMANCE:")
        for mode, losses in attention_stats.items():
            avg_loss = np.mean(losses)
            std_loss = np.std(losses)
            best_loss = min(losses)
            print(f"   {mode.upper()}: Avg {avg_loss:.4f}±{std_loss:.4f}, Best {best_loss:.4f} ({len(losses)} experiments)")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Grid Search for Decision Transformer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--results_dir", type=str, default="grid_search_results", help="Results directory")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_experiments", type=int, default=50, help="Maximum number of experiments")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 핵심 하이퍼파라미터 범위 설정
    parser.add_argument("--n_layers", nargs='+', type=int, default=[4, 6, 8], help="n_layers options (core parameter)")
    parser.add_argument("--attention_modes", nargs='+', type=str, default=["standard", "advanced", "hybrid"], help="attention mode options (core parameter)")
    
    # 고정 하이퍼파라미터 (필요시 변경 가능)
    parser.add_argument("--d_model", type=int, default=512, help="d_model (fixed)")
    parser.add_argument("--n_heads", type=int, default=8, help="n_heads (fixed)")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout (fixed)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate (fixed)")
    
    args = parser.parse_args()
    
    # 그리드 서치 설정
    config = GridSearchConfig(
        n_layers_options=args.n_layers,
        attention_mode_options=args.attention_modes,
        
        # 고정 파라미터
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_experiments=args.max_experiments,
        data_path=args.data_path,
        results_dir=args.results_dir,
        device=args.device
    )
    
    # 그리드 서치 실행
    grid_search = HyperparameterGridSearch(config)
    results = grid_search.run_grid_search()
    
    print(f"\n🎉 Grid search completed! Results saved to {config.results_dir}")


if __name__ == "__main__":
    main()
