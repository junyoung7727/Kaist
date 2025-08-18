#!/usr/bin/env python3
"""
scal_test_result 디렉토리의 파일들을 3개씩 묶어서 하나의 데이터 파일로 합치는 스크립트
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import datetime

def load_json_file(file_path: str) -> Dict[str, Any]:
    """JSON 파일을 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {file_path} - {e}")
        return {}

def merge_json_files(file_paths: List[str], output_path: str) -> bool:
    """여러 JSON 파일을 하나로 합칩니다."""
    merged_data = {
        "merged_timestamp": datetime.datetime.now().isoformat(),
        "source_files": file_paths,
        "merged_results": [],
        "merged_circuits": {},
        "total_experiments": 0
    }
    
    for file_path in file_paths:
        print(f"📄 처리 중: {os.path.basename(file_path)}")
        data = load_json_file(file_path)
        
        if not data:
            continue
            
        # 파일 타입 구분: circuit 파일인지 result 파일인지 확인
        is_circuit_file = "_circ.json" in file_path
        
        if is_circuit_file:
            # 회로 파일 처리
            if "circuits" in data:
                circuit_count = len(data["circuits"])
                print(f"   - 회로 스펙: {circuit_count}개")
                merged_data["merged_circuits"].update(data["circuits"])
            
            if "merged_circuits" in data:
                circuit_count = len(data["merged_circuits"])
                print(f"   - 시뮬레이터 형식 회로: {circuit_count}개")
                merged_data["merged_circuits"].update(data["merged_circuits"])
        else:
            # 결과 파일 처리
            if "results" in data:
                result_count = len(data["results"])
                print(f"   - IBM 형식 결과: {result_count}개")
                merged_data["merged_results"].extend(data["results"])
            
            else:
                result_count = len(data)
                print(f"   - 시뮬레이터 형식 결과: {result_count}개")
                merged_data["merged_results"].extend(data)
            
        # experiment_config 정보 (첫 번째 파일 기준)
        if "experiment_config" in data and "experiment_config" not in merged_data:
            merged_data["experiment_config"] = data["experiment_config"]
            
        # experiment_name 정보 (첫 번째 파일 기준)
        if "experiment_name" in data and "experiment_name" not in merged_data:
            merged_data["experiment_name"] = data["experiment_name"]
    
    merged_data["total_experiments"] = len(merged_data["merged_results"])
    
    # 디버깅: 결과와 회로 ID 매칭 확인
    result_ids = set(result["circuit_id"] for result in merged_data["merged_results"])
    circuit_ids = set(merged_data["merged_circuits"].keys())
    
    missing_circuits = result_ids - circuit_ids
    missing_results = circuit_ids - result_ids
    
    if missing_circuits:
        print(f"⚠️ 회로 스펙이 없는 실험: {len(missing_circuits)}개")
        print(f"   예시: {list(missing_circuits)[:5]}")
    
    if missing_results:
        print(f"⚠️ 실험 결과가 없는 회로: {len(missing_results)}개")
        print(f"   예시: {list(missing_results)[:5]}")
    
    # 결과 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 병합 완료: {output_path}")
        print(f"   - 총 실험 수: {merged_data['total_experiments']}")
        print(f"   - 총 서킷 수: {len(merged_data['merged_circuits'])}")
        print(f"   - 매칭된 회로: {len(result_ids & circuit_ids)}개")
        return True
    except Exception as e:
        print(f"❌ 저장 실패: {output_path} - {e}")
        return False

def main():
    """메인 실행 함수"""
    # scal_test_result 디렉토리 경로
    scal_dir = Path("c:/Users/jungh/Documents/GitHub/Kaist/scal_test_result")
    
    if not scal_dir.exists():
        print(f"❌ 디렉토리를 찾을 수 없습니다: {scal_dir}")
        return
    
    # JSON 파일들 찾기 - result 파일과 circuit 파일 분리
    result_files = list(scal_dir.glob("*_result.json"))
    circuit_files = list(scal_dir.glob("*_result_circ.json"))
    
    result_files.sort()  # 파일명 순으로 정렬
    circuit_files.sort()
    
    print(f"📁 발견된 결과 파일 수: {len(result_files)}")
    for file in result_files:
        print(f"   - {file.name}")
    
    print(f"📁 발견된 회로 파일 수: {len(circuit_files)}")
    for file in circuit_files:
        print(f"   - {file.name}")
    
    if len(result_files) == 0:
        print("❌ 결과 파일이 없습니다.")
        return
    
    # 모든 파일을 하나로 병합 (1,2,3번 파일 모두 함께)
    output_dir = scal_dir / "merged_results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🔄 전체 파일 병합 처리 중...")
    print(f"   결과 파일들: {[f.name for f in result_files]}")
    print(f"   회로 파일들: {[f.name for f in circuit_files]}")
    
    # 출력 파일명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"merged_all_{timestamp}.json"
    
    # 병합 실행 (결과 파일과 회로 파일 모두 포함)
    all_files = [str(f) for f in result_files] + [str(f) for f in circuit_files]
    if merge_json_files(all_files, str(output_file)):
        print(f"✅ 전체 병합 완료")
    else:
        print(f"❌ 병합 실패")
    
    print(f"\n🎉 전체 작업 완료!")
    print(f"   - 출력 디렉토리: {output_dir}")

if __name__ == "__main__":
    main()
