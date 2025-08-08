#!/usr/bin/env python3
"""
CUDA 및 GPU 사용 가능 여부 진단 스크립트
"""

import torch
import sys
import subprocess
import os

def check_cuda_availability():
    """CUDA 사용 가능 여부 체크"""
    print("=" * 60)
    print("CUDA 및 GPU 진단 시작")
    print("=" * 60)
    
    # 1. PyTorch 버전 확인
    print(f"PyTorch 버전: {torch.__version__}")
    
    # 2. CUDA 사용 가능 여부
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA를 사용할 수 없습니다!")
        print("\n가능한 원인들:")
        print("1. CUDA가 설치되지 않음")
        print("2. PyTorch가 CPU 버전으로 설치됨")
        print("3. NVIDIA 드라이버 문제")
        print("4. 환경 변수 설정 문제")
        return False
    
    # 3. CUDA 버전 정보
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
    
    # 4. GPU 개수 및 정보
    gpu_count = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 5. 현재 GPU 설정
    if gpu_count > 0:
        current_device = torch.cuda.current_device()
        print(f"현재 사용 중인 GPU: {torch.cuda.current_device()}")
    
    return True

def test_gpu_operations():
    """GPU 연산 테스트"""
    print("\n" + "=" * 60)
    print("GPU 연산 테스트")
    # GPU 연산 테스트
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없어 테스트를 건너뜁니다.")
        return False
    
    try:
        # 간단한 텐서 연산 테스트
        print("GPU 메모리 할당 테스트...")
        device = torch.device('cuda')
        
        # 작은 텐서로 시작
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        
        print("텐서 생성 성공")
        
        # 행렬 곱셈 테스트
        print("행렬 곱셈 테스트...")
        z = torch.matmul(x, y)
        print("행렬 곱셈 성공")
        
        # GPU 메모리 사용량 확인
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU 메모리 사용량: {allocated:.1f} MB (할당) / {cached:.1f} MB (캐시)")
        
        # 메모리 정리
        del x, y, z
        torch.cuda.empty_cache()
        print("GPU 메모리 정리 완료")
        
        return True
        
    except Exception as e:
        print(f"GPU 연산 테스트 실패: {e}")
        return False

def check_nvidia_driver():
    """NVIDIA 드라이버 확인"""
    print("\n" + "=" * 60)
    print("NVIDIA 드라이버 확인")
    print("=" * 60)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("nvidia-smi 실행 성공")
            print("\nGPU 정보:")
            print(result.stdout)
            return True
        else:
            print("nvidia-smi 실행 실패")
            print(f"에러: {result.stderr}")
            return False
    except FileNotFoundError:
        print("nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버가 설치되지 않았을 수 있습니다.")
        return False
    except subprocess.TimeoutExpired:
        print("nvidia-smi 실행 시간 초과")
        return False
    except Exception as e:
        print(f"nvidia-smi 실행 중 오류: {e}")
        return False

def check_environment():
    """환경 변수 확인"""
    print("\n" + "=" * 60)
    print("환경 변수 확인")
    print("=" * 60)
    
    cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT']
    for var in cuda_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var}: 설정되지 않음")
    
    # PATH에서 CUDA 확인
    path = os.environ.get('PATH', '')
    cuda_in_path = any('cuda' in p.lower() for p in path.split(os.pathsep))
    print(f"PATH에 CUDA 포함: {cuda_in_path}")

def provide_solutions():
    """해결 방법 제시"""
    print("\n" + "=" * 60)
    print("문제 해결 방법")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없는 경우 해결 방법:")
        print("\n1. PyTorch 재설치 (CUDA 버전):")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n2. NVIDIA 드라이버 설치/업데이트:")
        print("   https://www.nvidia.com/drivers 에서 최신 드라이버 다운로드")
        
        print("\n3. CUDA Toolkit 설치:")
        print("   https://developer.nvidia.com/cuda-downloads")
        
        print("\n4. 환경 변수 설정:")
        print("   CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8")
        print("   PATH에 %CUDA_PATH%\\bin 추가")

def main():
    """메인 함수"""
    print("CUDA 진단 스크립트 시작\n")
    
    # 1. CUDA 사용 가능 여부 확인
    cuda_ok = check_cuda_availability()
    
    # 2. NVIDIA 드라이버 확인
    driver_ok = check_nvidia_driver()
    
    # 3. 환경 변수 확인
    check_environment()
    
    # 4. GPU 연산 테스트 (CUDA 사용 가능한 경우)
    if cuda_ok:
        test_ok = test_gpu_operations()
    else:
        test_ok = False
    
    # 5. 결과 요약
    print("\n" + "=" * 60)
    print("진단 결과 요약")
    print("=" * 60)
    print(f"CUDA 사용 가능: {'OK' if cuda_ok else 'FAIL'}")
    print(f"NVIDIA 드라이버: {'OK' if driver_ok else 'FAIL'}")
    print(f"GPU 연산 테스트: {'OK' if test_ok else 'FAIL'}")
    
    if not (cuda_ok and driver_ok and test_ok):
        print("\n문제가 발견되었습니다.")
        provide_solutions()
    else:
        print("\n모든 테스트 통과! GPU를 사용할 수 있습니다.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
