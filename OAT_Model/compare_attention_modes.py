"""
Simple Attention Mode Comparison Script
단순한 어텐션 메커니즘 비교 스크립트 (하이퍼파라미터 그리드 서치는 hyperparameter_grid_search.py 사용)
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import argparse
from typing import Dict, Any, List
import time
import numpy as np
from dataclasses import asdict

# 프로젝트 모듈 임포트
from src.models.decision_transformer import DecisionTransformer
from src.training.trainer import TrainingConfig, QuantumCircuitCollator
from src.data.embedding_pipeline import EmbeddingPipeline, EmbeddingConfig
from src.data.quantum_circuit_dataset import DatasetManager, create_dataloaders
from utils.debug_utils import debug_print


class AttentionModeComparator:
    """어텐션 모드별 성능 비교 클래스"""
    
    def __init__(self, config: TrainingConfig, model_path: str = None):
        self.config = config
        self.device = torch.device(config.device)
        
        # 임베딩 파이프라인 설정
        embedding_config = EmbeddingConfig(
            d_model=config.d_model,
            n_gate_types=config.n_gate_types
        )
        self.embedding_pipeline = EmbeddingPipeline(embedding_config)
        
        # 모델 생성 (기본 모드로)
        self.model = DecisionTransformer(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_gate_types=config.n_gate_types,
            dropout=config.dropout,
            attention_mode="standard"  # 시작은 표준 모드
        ).to(self.device)
        
        # 모델 로딩
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")
        else:
            print("⚠️ No model loaded - using random weights")
    
    def load_model(self, model_path: str):
        """모델 체크포인트 로딩"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def compare_modes_on_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """단일 배치에서 어텐션 모드별 비교"""
        self.model.eval()
        results = {}
        
        modes = ["standard", "advanced", "hybrid"]
        
        with torch.no_grad():
            for mode in modes:
                print(f"🔄 Testing {mode} attention mode...")
                
                # 모드 변경
                self.model.set_attention_mode(mode)
                
                # 추론 시간 측정
                start_time = time.time()
                
                # 모델 실행
                output = self.model(
                    input_sequence=batch['input_sequence'],
                    attention_mask=batch['attention_mask'],
                    action_prediction_mask=batch['action_prediction_mask']
                )
                
                inference_time = time.time() - start_time
                
                # 결과 저장
                results[mode] = {
                    'action_logits': output['action_logits'].cpu(),
                    'action_predictions': output['action_predictions'].cpu(),
                    'hidden_states': output['hidden_states'].cpu(),
                    'inference_time': inference_time,
                    'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                }
                
                print(f"   ⏱️ Inference time: {inference_time:.4f}s")
                if torch.cuda.is_available():
                    print(f"   💾 Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        
        return results
    
    def analyze_differences(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """어텐션 모드별 차이점 분석"""
        analysis = {}
        
        # 성능 비교
        performance = {}
        for mode in results:
            performance[mode] = {
                'inference_time': results[mode]['inference_time'],
                'memory_usage': results[mode]['memory_usage']
            }
        analysis['performance'] = performance
        
        # 출력 차이 분석
        if len(results) >= 2:
            modes = list(results.keys())
            base_mode = modes[0]
            
            differences = {}
            for mode in modes[1:]:
                # 액션 로짓 차이
                logits_diff = torch.abs(
                    results[mode]['action_logits'] - results[base_mode]['action_logits']
                ).mean().item()
                
                # 히든 스테이트 차이
                hidden_diff = torch.abs(
                    results[mode]['hidden_states'] - results[base_mode]['hidden_states']
                ).mean().item()
                
                differences[f"{base_mode}_vs_{mode}"] = {
                    'logits_difference': logits_diff,
                    'hidden_difference': hidden_diff
                }
            
            analysis['output_differences'] = differences
        
        return analysis
    
    def run_comparison_experiment(self, data_path: str, num_batches: int = 5) -> Dict[str, Any]:
        """전체 비교 실험 실행"""
        print(f"🚀 Starting attention mode comparison experiment...")
        print(f"   📁 Data path: {data_path}")
        print(f"   🔢 Number of batches: {num_batches}")
        
        # 데이터 로더 생성
        dataset_manager = DatasetManager(data_path)
        train_loader, val_loader = create_dataloaders(
            dataset_manager=dataset_manager,
            embedding_pipeline=self.embedding_pipeline,
            batch_size=self.config.batch_size,
            train_split=0.8
        )
        
        all_results = []
        all_analyses = []
        
        # 여러 배치에서 실험
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
                
            print(f"\n📊 Batch {i+1}/{num_batches}")
            
            # 배치를 디바이스로 이동
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 모드별 비교
            batch_results = self.compare_modes_on_batch(batch)
            analysis = self.analyze_differences(batch_results)
            
            all_results.append(batch_results)
            all_analyses.append(analysis)
        
        # 전체 결과 집계
        summary = self.summarize_results(all_analyses)
        
        return {
            'config': asdict(self.config),
            'batch_results': all_results,
            'batch_analyses': all_analyses,
            'summary': summary
        }
    
    def summarize_results(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 요약"""
        summary = {}
        
        # 성능 평균
        if analyses:
            modes = list(analyses[0]['performance'].keys())
            avg_performance = {}
            
            for mode in modes:
                times = [a['performance'][mode]['inference_time'] for a in analyses]
                memories = [a['performance'][mode]['memory_usage'] for a in analyses]
                
                avg_performance[mode] = {
                    'avg_inference_time': np.mean(times),
                    'std_inference_time': np.std(times),
                    'avg_memory_usage': np.mean(memories),
                    'std_memory_usage': np.std(memories)
                }
            
            summary['average_performance'] = avg_performance
            
            # 출력 차이 평균
            if 'output_differences' in analyses[0]:
                diff_keys = list(analyses[0]['output_differences'].keys())
                avg_differences = {}
                
                for key in diff_keys:
                    logits_diffs = [a['output_differences'][key]['logits_difference'] for a in analyses]
                    hidden_diffs = [a['output_differences'][key]['hidden_difference'] for a in analyses]
                    
                    avg_differences[key] = {
                        'avg_logits_difference': np.mean(logits_diffs),
                        'std_logits_difference': np.std(logits_diffs),
                        'avg_hidden_difference': np.mean(hidden_diffs),
                        'std_hidden_difference': np.std(hidden_diffs)
                    }
                
                summary['average_differences'] = avg_differences
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """결과 저장"""
        # 텐서를 리스트로 변환 (JSON 직렬화를 위해)
        def tensor_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_list(item) for item in obj]
            else:
                return obj
        
        serializable_results = tensor_to_list(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare attention modes in Decision Transformer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--output_path", type=str, default="attention_comparison_results.json", 
                       help="Output path for results")
    parser.add_argument("--num_batches", type=int, default=5, help="Number of batches to test")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # 설정
    config = TrainingConfig(
        batch_size=args.batch_size,
        device=args.device,
        attention_mode="standard"  # 시작 모드
    )
    
    # 비교 실험 실행
    comparator = AttentionModeComparator(config, args.model_path)
    results = comparator.run_comparison_experiment(args.data_path, args.num_batches)
    
    # 결과 저장
    comparator.save_results(results, args.output_path)
    
    # 요약 출력
    print("\n" + "="*50)
    print("📈 EXPERIMENT SUMMARY")
    print("="*50)
    
    if 'summary' in results and 'average_performance' in results['summary']:
        perf = results['summary']['average_performance']
        for mode, stats in perf.items():
            print(f"\n🔧 {mode.upper()} Attention:")
            print(f"   ⏱️ Avg inference time: {stats['avg_inference_time']:.4f}±{stats['std_inference_time']:.4f}s")
            if stats['avg_memory_usage'] > 0:
                print(f"   💾 Avg memory usage: {stats['avg_memory_usage']/1024**2:.1f}±{stats['std_memory_usage']/1024**2:.1f}MB")
    
    if 'summary' in results and 'average_differences' in results['summary']:
        diffs = results['summary']['average_differences']
        print(f"\n🔍 OUTPUT DIFFERENCES:")
        for comparison, stats in diffs.items():
            print(f"   {comparison}:")
            print(f"     Logits diff: {stats['avg_logits_difference']:.6f}±{stats['std_logits_difference']:.6f}")
            print(f"     Hidden diff: {stats['avg_hidden_difference']:.6f}±{stats['std_hidden_difference']:.6f}")


if __name__ == "__main__":
    main()
