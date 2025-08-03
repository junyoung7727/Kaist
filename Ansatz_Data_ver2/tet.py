# CUDA-GPU 호환성 상세 진단
import subprocess
import sys
import pkg_resources

def check_cuda_versions():
    """다양한 CUDA 버전 확인"""
    print("=== CUDA 버전 호환성 확인 ===\n")
    
    # 1. nvidia-smi의 CUDA 버전
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            import re
            cuda_match = re.search(r'CUDA Version: ([\d.]+)', result.stdout)
            if cuda_match:
                nvidia_smi_cuda = cuda_match.group(1)
                print(f"1. nvidia-smi CUDA Version: {nvidia_smi_cuda}")
    except Exception as e:
        print(f"1. nvidia-smi 실패: {e}")
    
    # 2. nvcc 버전 (설치되어 있다면)
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"2. nvcc 출력:\n{result.stdout}")
        else:
            print("2. nvcc 없음 (정상 - WSL2에서는 필수 아님)")
    except FileNotFoundError:
        print("2. nvcc 없음 (정상 - WSL2에서는 필수 아님)")
    
    # 3. Python CUDA 패키지들
    print("\n3. Python CUDA 패키지 버전:")
    cuda_packages = [
        'nvidia-cuda-runtime-cu11',
        'nvidia-cublas-cu11', 
        'nvidia-cusolver-cu11',
        'nvidia-cusparse-cu11',
        'cuquantum-cu11'
    ]
    
    for pkg in cuda_packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"   {pkg}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"   {pkg}: 설치되지 않음")
    
    # 4. qiskit-aer 버전 및 빌드 정보
    print("\n4. Qiskit Aer 정보:")
    try:
        import qiskit_aer
        print(f"   qiskit-aer 버전: {qiskit_aer.__version__}")
        
        # AerSimulator의 빌드 정보 확인
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        config = sim.configuration()
        
        print(f"   Backend 이름: {config.backend_name}")
        print(f"   Backend 버전: {config.backend_version}")
        
        # 사용 가능한 방법들 확인
        methods = sim.available_methods()
        devices = sim.available_devices()
        print(f"   사용 가능한 방법: {methods}")
        print(f"   사용 가능한 디바이스: {devices}")
        
    except Exception as e:
        print(f"   Qiskit Aer 정보 확인 실패: {e}")

def check_compute_capability_support():
    """패키지가 지원하는 Compute Capability 확인"""
    print("\n=== Compute Capability 지원 확인 ===")
    
    # GTX 1060은 CC 6.1
    target_cc = (6, 1)
    print(f"타겟 GPU: GTX 1060 (CC {target_cc[0]}.{target_cc[1]})")
    
    try:
        # qiskit-aer-gpu 패키지 정보 확인
        import pkg_resources
        
        # 설치된 패키지 목록에서 qiskit-aer 관련 찾기
        for dist in pkg_resources.working_set:
            if 'qiskit-aer' in dist.project_name.lower():
                print(f"\n설치된 패키지: {dist.project_name} {dist.version}")
                print(f"설치 위치: {dist.location}")
                
                # 패키지 메타데이터 확인
                try:
                    metadata = dist.get_metadata_lines('METADATA')
                    for line in metadata:
                        if 'cuda' in line.lower() or 'compute' in line.lower():
                            print(f"메타데이터: {line}")
                except:
                    pass
    
    except Exception as e:
        print(f"패키지 정보 확인 실패: {e}")

def test_specific_cuda_runtime():
    """특정 CUDA 런타임 테스트"""
    print("\n=== CUDA 런타임 호환성 테스트 ===")
    
    try:
        # CUDA 런타임 정보
        import nvidia.cuda_runtime
        print("✓ nvidia.cuda_runtime 사용 가능")
        
        # 간단한 CUDA 호출 테스트
        try:
            # CUDA 디바이스 개수 확인
            print("CUDA 런타임 테스트 중...")
            
            # PyTorch로 CUDA 테스트 (설치되어 있다면)
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"✓ PyTorch CUDA 사용 가능")
                    print(f"  디바이스 개수: {torch.cuda.device_count()}")
                    print(f"  현재 디바이스: {torch.cuda.current_device()}")
                    print(f"  디바이스 이름: {torch.cuda.get_device_name()}")
                    
                    # 간단한 CUDA 연산 테스트
                    x = torch.randn(10, device='cuda')
                    y = x * 2
                    print("✓ PyTorch CUDA 연산 성공")
                else:
                    print("✗ PyTorch CUDA 사용 불가")
            except ImportError:
                print("PyTorch 설치되지 않음")
                
        except Exception as e:
            print(f"✗ CUDA 런타임 테스트 실패: {e}")
            
    except ImportError:
        print("✗ nvidia.cuda_runtime 없음")

def recommend_specific_solutions():
    """GTX 1060 CC 6.1 전용 해결책"""
    print("\n=== GTX 1060 CC 6.1 전용 해결책 ===")
    
    print("1. 다른 CUDA 버전 패키지 시도:")
    print("   # 현재 패키지 제거")
    print("   pip uninstall qiskit-aer-gpu-cu11 qiskit-aer -y")
    print() 
    print("   # CUDA 10.2 버전 시도 (더 넓은 CC 지원)")
    print("   pip install qiskit-aer-gpu-cu102  # 있다면")
    print()
    print("   # 또는 일반 CUDA 12 버전")
    print("   pip install qiskit-aer-gpu")
    print()
    
    print("2. 환경 변수로 CC 강제 지정:")
    print("   export CUDA_ARCH_LIST=\"6.1\"")
    print("   export TORCH_CUDA_ARCH_LIST=\"6.1\"")
    print()
    
    print("3. 소스에서 CC 6.1 타겟으로 컴파일:")
    print("   git clone https://github.com/Qiskit/qiskit-aer.git")
    print("   cd qiskit-aer")
    print("   CMAKE_ARGS=\"-DCUDA_ARCH_LIST=6.1\" pip install .")
    print()
    
    print("4. CUDA 드라이버/런타임 다운그레이드:")
    print("   # CUDA 11.4가 문제일 수 있음")
    print("   # Windows에서 CUDA 11.2 드라이버 설치")
    print()
    
    print("5. 최후 수단 - CPU 최적화 사용:")
    print("   pip uninstall qiskit-aer-gpu-cu11 -y")
    print("   pip install qiskit-aer")
    print("   # CPU도 GTX 1060보다 빠를 수 있음")

def main():
    print("CUDA-GPU 호환성 상세 진단")
    print("=" * 50)
    
    check_cuda_versions()
    check_compute_capability_support()
    test_specific_cuda_runtime()
    recommend_specific_solutions()
    
    print("\n=== 결론 ===")
    print("GTX 1060 (CC 6.1)은 지원되어야 하지만,")
    print("qiskit-aer-gpu-cu11 패키지가 CC 6.1을 포함하지 않을 수 있음")
    print("→ 다른 CUDA 버전 또는 소스 컴파일 필요")

if __name__ == "__main__":
    main()