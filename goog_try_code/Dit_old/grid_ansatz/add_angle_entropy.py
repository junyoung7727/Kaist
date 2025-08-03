#!/usr/bin/env python3
"""
각도 엔트로피 계산 및 추가 스크립트
- 기존 배치 결과에 각도 엔트로피 데이터 추가
- 기존 측정 데이터를 사용하여 계산
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from expressibility_calculator import calculate_angle_entropy

def load_batch_results(file_path):
    """배치 결과 파일 로드"""
    print(f"📂 배치 결과 파일 로드 중: {file_path}")
    with open(file_path, 'r') as f:
        batch_results = json.load(f)
    
    print(f"✅ 배치 결과 로드 완료: {len(batch_results)} 회로 데이터")
    return batch_results

def save_batch_results(batch_results, file_path):
    """배치 결과 파일 저장"""
    print(f"💾 배치 결과 파일 저장 중: {file_path}")
    with open(file_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"✅ 배치 결과 저장 완료: {file_path}")
    return file_path

def extract_vectors_from_measurements(result):
    """측정 결과에서 벡터 추출"""
    vectors = []
    weights = []
    
    # 기존 형식 검사: "measurements" -> "counts"
    measurements = result.get("measurements", {})
    if measurements:
        counts = measurements.get("counts", {})
        if counts:
            # 총 측정 횟수 계산
            total_counts = sum(counts.values())
            if total_counts > 0:
                # 비트열을 정수 인덱스로 변환하여 확률 분포 벡터 생성
                n_qubits = len(next(iter(counts.keys()))) if counts else 0
                # 메모리 제한으로 인해 너무 큰 배열을 만들지 않도록 제한
                if n_qubits > 0 and n_qubits <= 16:  # 16 큐비트로 제한 (2^16 = 65536)
                    try:
                        vector = np.zeros(2**n_qubits)
                        for bitstring, count in counts.items():
                            # 비트열을 정수로 변환 (예: '101' -> 5)
                            try:
                                idx = int(bitstring, 2)
                                vector[idx] = count / total_counts
                            except (ValueError, IndexError):
                                continue
                        vectors.append(vector)
                        weights.append(1.0)
                        return vectors, weights
                    except ValueError as e:
                        print(f"큐비트 수가 너무 많음 (n_qubits={n_qubits}): {str(e)}")
                else:
                    print(f"큐비트 수가 제한을 초과함 (n_qubits={n_qubits}). 최대 16까지 허용됩니다.")
    
    # 새로운 형식 검사: "execution_result" -> "measurement_counts"
    execution_result = result.get("execution_result", {})
    if execution_result:
        counts = execution_result.get("measurement_counts", {})
        if counts:
            # 총 측정 횟수 계산
            total_counts = sum(counts.values())
            if total_counts > 0:
                # 비트열을 정수 인덱스로 변환하여 확률 분포 벡터 생성
                n_qubits = len(next(iter(counts.keys()))) if counts else 0
                # 메모리 제한으로 인해 너무 큰 배열을 만들지 않도록 제한
                if n_qubits > 0 and n_qubits <= 16:  # 16 큐비트로 제한 (2^16 = 65536)
                    try:
                        vector = np.zeros(2**n_qubits)
                        for bitstring, count in counts.items():
                            # 비트열을 정수로 변환 (예: '101' -> 5)
                            try:
                                idx = int(bitstring, 2)
                                vector[idx] = count / total_counts
                            except (ValueError, IndexError):
                                continue
                        vectors.append(vector)
                        weights.append(1.0)
                    except ValueError as e:
                        print(f"큐비트 수가 너무 많음 (n_qubits={n_qubits}): {str(e)}")
                else:
                    print(f"큐비트 수가 제한을 초과함 (n_qubits={n_qubits}). 최대 16까지 허용됩니다.")
    
    return vectors, weights

def calculate_and_add_angle_entropy(batch_results):
    """각 회로에 대한 각도 엔트로피 계산 및 추가"""
    print("🔄 각도 엔트로피 계산 중...")
    
    # 히스토그램 구간 수 설정
    n_bins = 20
    
    # 각 회로에 대해 각도 엔트로피 계산
    for i, result in enumerate(tqdm(batch_results, desc="각도 엔트로피 계산")):
        # 측정 결과에서 벡터 추출
        vectors, weights = extract_vectors_from_measurements(result)
        
        # 각도 엔트로피 계산
        if vectors and len(vectors) >= 2:
            try:
                angle_entropy = calculate_angle_entropy(vectors, weights, n_bins)
                
                # 결과에 각도 엔트로피 추가
                # 기존 distribution entropy가 있는지 확인
                if "expressibility" in result and "entropy_based" in result["expressibility"]:
                    # 기존 entropy_based 섹션에 angle_entropy 추가
                    result["expressibility"]["entropy_based"]["angle_entropy"] = angle_entropy
                    result["expressibility"]["entropy_based"]["angle_entropy_calculation_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result["expressibility"]["entropy_based"]["angle_entropy_n_bins"] = n_bins
                    result["expressibility"]["entropy_based"]["angle_entropy_n_vectors"] = len(vectors)
                else:
                    # expressibility.entropy_based 섹션이 없는 경우 새로 생성
                    if "expressibility" not in result:
                        result["expressibility"] = {}
                    result["expressibility"]["entropy_based"] = {
                        "angle_entropy": angle_entropy,
                        "angle_entropy_calculation_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "angle_entropy_n_bins": n_bins,
                        "angle_entropy_n_vectors": len(vectors)
                    }
                
                # 호환성을 위해 기존 위치에도 유지 (추후 제거 예정)
                result["angle_entropy"] = angle_entropy
                result["angle_entropy_calculation"] = {
                    "n_bins": n_bins,
                    "n_vectors": len(vectors),
                    "calculation_time": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
            except Exception as e:
                print(f"각도 엔트로피 계산 중 오류 발생 (회로 {i}): {str(e)}")
                # 에러 정보 추가
                if "expressibility" in result and "entropy_based" in result["expressibility"]:
                    result["expressibility"]["entropy_based"]["angle_entropy"] = None
                    result["expressibility"]["entropy_based"]["angle_entropy_error"] = str(e)
                    result["expressibility"]["entropy_based"]["angle_entropy_calculation_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                else:
                    if "expressibility" not in result:
                        result["expressibility"] = {}
                    result["expressibility"]["entropy_based"] = {
                        "angle_entropy": None,
                        "angle_entropy_error": str(e),
                        "angle_entropy_calculation_time": datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                
                # 호환성을 위해 기존 위치에도 유지
                result["angle_entropy"] = None
                result["angle_entropy_calculation"] = {
                    "error": str(e),
                    "calculation_time": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
        else:
                # 측정 결과가 없는 경우, 다른 측정 결과 형식 확인
            # 일부 결과는 다른 형식으로 저장되어 있을 수 있음
            try:
                # 다른 형식의 측정 결과 확인
                if "expressibility_result" in result and "estimated_moments" in result["expressibility_result"]:
                    # expressibility_result에서 모멘트 추출하여 벡터로 사용
                    moments = result["expressibility_result"]["estimated_moments"]
                    if isinstance(moments, list) and len(moments) >= 2:
                        try:
                            vectors = [np.array(moment) for moment in moments]
                            weights = [1.0] * len(vectors)
                            angle_entropy = calculate_angle_entropy(vectors, weights, n_bins)
                        except ValueError as e:
                            print(f"모멘트 데이터 처리 중 오류 발생: {str(e)}")
                            raise e
                        
                        # expressibility.entropy_based 섹션에 angle_entropy 추가
                        if "expressibility" in result and "entropy_based" in result["expressibility"]:
                            result["expressibility"]["entropy_based"]["angle_entropy"] = angle_entropy
                            result["expressibility"]["entropy_based"]["angle_entropy_n_bins"] = n_bins
                            result["expressibility"]["entropy_based"]["angle_entropy_n_vectors"] = len(vectors)
                            result["expressibility"]["entropy_based"]["angle_entropy_source"] = "expressibility_moments"
                            result["expressibility"]["entropy_based"]["angle_entropy_calculation_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                        else:
                            if "expressibility" not in result:
                                result["expressibility"] = {}
                            result["expressibility"]["entropy_based"] = {
                                "angle_entropy": angle_entropy,
                                "angle_entropy_n_bins": n_bins,
                                "angle_entropy_n_vectors": len(vectors),
                                "angle_entropy_source": "expressibility_moments",
                                "angle_entropy_calculation_time": datetime.now().strftime("%Y%m%d_%H%M%S")
                            }
                        
                        # 호환성을 위해 기존 위치에도 유지
                        result["angle_entropy"] = angle_entropy
                        result["angle_entropy_calculation"] = {
                            "n_bins": n_bins,
                            "n_vectors": len(vectors),
                            "source": "expressibility_moments",
                            "calculation_time": datetime.now().strftime("%Y%m%d_%H%M%S")
                        }
                    else:
                        raise ValueError("모멘트 데이터 부족")
                else:
                    raise ValueError("측정 결과 없음")
            except Exception as e:
                print(f"회로 {i}에 대한 각도 엔트로피 계산 불가: {str(e)}")
                # 에러 정보 추가
                if "expressibility" in result and "entropy_based" in result["expressibility"]:
                    result["expressibility"]["entropy_based"]["angle_entropy"] = None
                    result["expressibility"]["entropy_based"]["angle_entropy_error"] = f"데이터 부족: {str(e)}"
                    result["expressibility"]["entropy_based"]["angle_entropy_calculation_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                else:
                    if "expressibility" not in result:
                        result["expressibility"] = {}
                    result["expressibility"]["entropy_based"] = {
                        "angle_entropy": None,
                        "angle_entropy_error": f"데이터 부족: {str(e)}",
                        "angle_entropy_calculation_time": datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                
                # 호환성을 위해 기존 위치에도 유지
                result["angle_entropy"] = None
                result["angle_entropy_calculation"] = {
                    "error": f"데이터 부족: {str(e)}",
                    "calculation_time": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
    
    print(f"✅ 각도 엔트로피 계산 완료!")
    return batch_results

def main():
    """메인 함수"""
    print("🚀 각도 엔트로피 계산 및 추가 스크립트 시작")
    
    # 배치 결과 파일 경로
    batch_file = "grid_circuits/mega_results/batch_1_results_20250529_101750.json"
    
    # 배치 결과 로드
    batch_results = load_batch_results(batch_file)
    
    # 각도 엔트로피 계산 및 추가
    updated_results = calculate_and_add_angle_entropy(batch_results)
    
    # 업데이트된 결과 저장 (새 파일로)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"grid_circuits/mega_results/batch_1_results_20250529_101750_with_angle_entropy_{timestamp}.json"
    save_batch_results(updated_results, output_file)
    
    # 각도 엔트로피 통계 출력
    entropy_values = [r.get("angle_entropy") for r in updated_results if r.get("angle_entropy") is not None]
    if entropy_values:
        print(f"\n📊 각도 엔트로피 통계:")
        print(f"  - 계산된 회로 수: {len(entropy_values)}/{len(updated_results)}")
        print(f"  - 평균: {np.mean(entropy_values):.4f}")
        print(f"  - 최소: {np.min(entropy_values):.4f}")
        print(f"  - 최대: {np.max(entropy_values):.4f}")
        print(f"  - 표준편차: {np.std(entropy_values):.4f}")
    
    print(f"\n🎉 작업 완료! 업데이트된 결과가 저장되었습니다: {output_file}")

if __name__ == "__main__":
    main()
