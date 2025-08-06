@echo off
echo ============================================================
echo 🧬 Qiskit 양자 알고리즘 라이브러리 설치
echo ============================================================

echo.
echo 📦 기본 Qiskit 설치...
pip install qiskit

echo.
echo 🧪 Qiskit Algorithms (VQE, QAOA 등)...
pip install qiskit-algorithms

echo.
echo 🧬 Qiskit Nature (분자 시뮬레이션, VQE)...
pip install qiskit-nature

echo.
echo 🎯 Qiskit Optimization (QAOA, 최적화 문제)...
pip install qiskit-optimization

echo.
echo 📊 추가 유용한 패키지들...
pip install matplotlib networkx

echo.
echo ✅ 설치 완료!
echo.
echo 🧪 VQE 예제 실행:
echo python vqe_example.py
echo.
echo 🎯 QAOA 예제 실행:
echo python qaoa_example.py
echo.
pause
