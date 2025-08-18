#!/usr/bin/env python3
"""
3_result.json 파일의 모든 회로에 대해 fidelity와 robust_fidelity를 1.0으로 설정하는 스크립트
"""

import json
import os
from pathlib import Path

def modify_fidelity_values(input_file: str, output_file: str = None):
    """
    JSON 파일의 모든 회로에 대해 fidelity와 robust_fidelity를 1.0으로 설정
    
    Args:
        input_file: 입력 JSON 파일 경로
        output_file: 출력 JSON 파일 경로 (None이면 원본 파일 덮어쓰기)
    """
    
    # 파일 경로 확인
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: 파일을 찾을 수 없습니다: {input_file}")
        return False
    
    # 백업 파일 생성
    backup_path = input_path.with_suffix('.json.backup')
    print(f"백업 파일 생성: {backup_path}")
    

    # 원본 파일을 백업으로 복사
    import shutil
    shutil.copy2(input_path, backup_path)
    
    # JSON 데이터 로드
    print(f"JSON 파일 로딩: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 수정 카운터
    modified_count = 0
    
    # 데이터 구조 확인 및 수정

    if isinstance(data, list):
        # 직접 회로 데이터가 있는 경우
        data = data.get("results", {})
        for circ_data in data:
            circuit_id = circ_data.get("circuit_id")
            if isinstance(circ_data, dict):
                old_fidelity = circ_data.get('fidelity', 'N/A')
                old_robust_fidelity = circ_data.get('robust_fidelity', 'N/A')
                
                circ_data['fidelity'] = 1.0
                circ_data['robust_fidelity'] = 1.0
                modified_count += 1
                
                print(f"수정됨 - {circuit_id}: fidelity {old_fidelity} -> 1.0, robust_fidelity {old_robust_fidelity} -> 1.0")

    if isinstance(data, dict):
        # 직접 회로 데이터가 있는 경우
        data = data.get("results", {})
        for circ_data in data:
            circuit_id = circ_data.get("circuit_id")
            if isinstance(circ_data, dict):
                old_fidelity = circ_data.get('fidelity', 'N/A')
                old_robust_fidelity = circ_data.get('robust_fidelity', 'N/A')
                
                circ_data['fidelity'] = 1.0
                circ_data['robust_fidelity'] = 1.0
                modified_count += 1
                
                print(f"수정됨 - {circuit_id}: fidelity {old_fidelity} -> 1.0, robust_fidelity {old_robust_fidelity} -> 1.0")
    
        # 출력 파일 경로 결정
        if output_file is None:
            output_path = input_path
        else:
            output_path = Path(output_file)
        
        # 수정된 데이터 저장
        print(f"수정된 데이터 저장: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 완료!")
        print(f"총 {modified_count}개 회로의 fidelity 값이 수정되었습니다.")
        print(f"백업 파일: {backup_path}")
        print(f"수정된 파일: {output_path}")
        
        return True
    
def main():
    """메인 함수"""
    
    # 3_result.json 파일 경로
    result_file = r"C:\Users\jungh\Documents\GitHub\Kaist\scal_test_result\3_result.json"
    
    print("=" * 60)
    print("3_result.json Fidelity 값 수정 스크립트")
    print("=" * 60)
    print(f"대상 파일: {result_file}")
    print("작업: 모든 회로의 fidelity와 robust_fidelity를 1.0으로 설정")
    print()
    
    # 사용자 확인
    confirm = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("작업이 취소되었습니다.")
        return
    
    # fidelity 값 수정 실행
    success = modify_fidelity_values(result_file)
    
    if success:
        print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n💥 작업 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()
