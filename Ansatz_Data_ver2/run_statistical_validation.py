#!/usr/bin/env python3
"""
통계적 검증 실행 스크립트

다양한 양자 지표에 대한 통계적 검증을 실행하는 메인 스크립트입니다.
"""

import sys
import argparse
from pathlib import Path
from config import Exp_Box

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent / "core"))

from core.statistical_validation_framework import (
    validate_entanglement, 
    validate_expressibility,
)


def main():
    """메인 실행 함수 - ExperimentConfig 직접 사용"""
    
    # 출력 디렉토리 생성
    output_dir = Path('validation_results')
    output_dir.mkdir(exist_ok=True)
    
    # 실험 설정 사용
    
    exp_config = Exp_Box.statistical_validation_config
    from execution.executor import QuantumExecutorFactory
    exp_config.executor = QuantumExecutorFactory.create_executor('ibm')
    num_repetitions = 3
    
    print("🚀 Meyer-Wallach Entanglement 통계적 검증 시작")
    print(f"   📊 큐빗 수: {exp_config.num_qubits}")
    print(f"   📏 회로 깊이: {exp_config.depth}")
    print(f"   🔄 회로 수: {exp_config.num_circuits}")
    print(f"   🔁 반복 횟수: {num_repetitions}")
    print(f"   📁 출력 디렉토리: {output_dir}")
    print()
    
    results = {}
    metric = 'all'
    save_plots = True
    # Meyer-Wallach Entanglement 검증
    print("=" * 60)
    print("🔗 MEYER-WALLACH ENTANGLEMENT 검증")
    print("=" * 60)
    
    if metric in ['entanglement', 'all']:
        entanglement_results = validate_entanglement(
            exp_config=exp_config,
            num_repetitions=num_repetitions,
            save_path=str(output_dir / 'entanglement_validation.png') if save_plots else None
        )
        results['entanglement'] = entanglement_results
        print(f"✅ Entanglement 검증 완료: {len(entanglement_results)}개 결과")
    
    # Expressibility 검증
    if metric in ['expressibility', 'all']:
        expressibility_results = validate_expressibility(
            exp_config=exp_config,
            num_repetitions=num_repetitions,
            save_path=str(output_dir / 'expressibility_validation.png') if save_plots else None
        )
        results['expressibility'] = expressibility_results
        print(f"✅ Expressibility 검증 완료: {len(expressibility_results)}개 결과")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 검증 결과 요약")
    print("=" * 60)
    
    if 'entanglement' in results and results['entanglement']:
        total_measurements = sum(len(r.measured_values) for r in results['entanglement'])
        avg_rmse = sum(r.statistics['rmse'] for r in results['entanglement']) / len(results['entanglement'])
    if entanglement_results:
        total_measurements = sum(len(r.measured_values) for r in entanglement_results)
        avg_rmse = sum(r.statistics['rmse'] for r in entanglement_results) / len(entanglement_results)
        
        print(f"🎯 ENTANGLEMENT:")
        print(f"   • 검증된 회로: {len(entanglement_results)}개")
        print(f"   • 총 측정 횟수: {total_measurements}회")
        print(f"   • 평균 RMSE: {avg_rmse:.6f}")
        print(f"   • 결과 저장: {output_dir}/entanglement_validation.png")

    if 'expressibility' in results and results['expressibility']:
        total_measurements = sum(len(r.measured_values) for r in results['expressibility'])
        avg_rmse = sum(r.statistics['rmse'] for r in results['expressibility']) / len(results['expressibility'])
    
        print(f"🎯 EXPRESSIBILITY:")
        print(f"   • 검증된 회로: {len(expressibility_results)}개")
        print(f"   • 총 측정 횟수: {total_measurements}회")
        print(f"   • 평균 RMSE: {avg_rmse:.6f}")
        print(f"   • 결과 저장: {output_dir}/expressibility_validation.png")

    print(f"\n🎉 검증이 완료되었습니다!")
    print(f"📁 결과는 {output_dir} 디렉토리에 저장되었습니다.")
    
    return results


def quick_purity_validation():
    """빠른 purity 검증 (개발/테스트용)"""
    print("🚀 빠른 Purity 검증 실행")
    
    # 간단한 설정
    exp_config = Exp_Box.statistical_validation_config
    
    results = validate_entanglement(
        exp_config=exp_config,
        num_repetitions=2,
        save_path='quick_entanglement_test.png'
    )
    
    print(f"✅ 빠른 검증 완료: {len(results)}개 결과")
    return results


def comprehensive_validation():
    """포괄적인 검증 (연구용)"""
    print("🚀 포괄적인 검증 실행")
    
    results = {}
    
    # 모든 지표 검증
    for metric_name, validate_func in [
        ('entanglement', validate_entanglement),
        ('expressibility', validate_expressibility)
    ]:
        print(f"\n📊 {metric_name.upper()} 검증 중...")
        
        try:
            metric_results = validate_func(
                exp_config=exp_config,
                num_repetitions=5,
                save_path=f'comprehensive_{metric_name}_validation.png'
            )
            results[metric_name] = metric_results
            print(f"✅ {metric_name} 완료")
        except Exception as e:
            print(f"❌ {metric_name} 실패: {str(e)}")
    
    return results


if __name__ == "__main__":
    main()

