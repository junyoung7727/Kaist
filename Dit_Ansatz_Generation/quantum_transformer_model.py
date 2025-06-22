#!/usr/bin/env python3
"""
양자 회로 표현력 및 피델리티 예측을 위한 트랜스포머 모델
우리의 mega job 데이터 구조에 최적화됨
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import gzip
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 설정 파라미터
MAX_SEQUENCE_LENGTH = 100  # 우리 데이터의 최대 시퀀스 길이
LATENT_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################
# 1. 우리 데이터 구조에 맞는 데이터셋
#################################################

class QuantumMegaJobDataset(Dataset):
    def __init__(self, data_dir, split='train', train_ratio=0.8):
        """
        Mega Job 데이터셋 로드
        data_dir: training_data 디렉토리 경로
        """
        self.data_dir = data_dir
        self.split = split
        
        # 배치 파일들 로드
        self.batch_files = [f for f in os.listdir(data_dir) 
                           if f.startswith('mega_batch_') and f.endswith('.json.gz')]
        
        # 훈련/검증 분할
        np.random.shuffle(self.batch_files)
        split_idx = int(len(self.batch_files) * train_ratio)
        
        if split == 'train':
            self.batch_files = self.batch_files[:split_idx]
        else:
            self.batch_files = self.batch_files[split_idx:]
        
        # 모든 회로 데이터 로드
        self.circuits = []
        self._load_all_circuits()
        
        print(f"{split} 데이터셋: {len(self.circuits)}개 회로 로드됨")
    
    def _load_all_circuits(self):
        """모든 배치 파일에서 회로 데이터 로드"""
        for batch_file in tqdm(self.batch_files, desc=f"{self.split} 데이터 로드 중"):
            batch_path = os.path.join(self.data_dir, batch_file)
            
            try:
                with gzip.open(batch_path, 'rt') as f:
                    batch_data = json.load(f)
                
                # 각 회로 데이터 추출
                for circuit in batch_data['circuits']:
                    self.circuits.append(circuit)
                    
            except Exception as e:
                print(f"배치 파일 {batch_file} 로드 실패: {str(e)}")
    
    def __len__(self):
        return len(self.circuits)
    
    def __getitem__(self, idx):
        circuit = self.circuits[idx]
        
        # 트랜스포머 입력 데이터 추출
        transformer_input = circuit.get('transformer_input', {})
        
        # 시퀀스 데이터 (패딩 처리됨)
        gate_sequence = torch.LongTensor(transformer_input.get('gate_sequence', [0] * MAX_SEQUENCE_LENGTH))
        qubit_sequence = torch.LongTensor(transformer_input.get('qubit_sequence', [-1] * MAX_SEQUENCE_LENGTH))
        param_sequence = torch.FloatTensor(transformer_input.get('param_sequence', [0.0] * MAX_SEQUENCE_LENGTH))
        gate_type_sequence = torch.LongTensor(transformer_input.get('gate_type_sequence', [0] * MAX_SEQUENCE_LENGTH))
        
        # 특성 데이터
        features = circuit.get('features', {})
        
        # 구조적 특성
        structural_features = torch.FloatTensor([
            features.get('n_qubits', 0) / 127.0,  # 정규화 (최대 127큐빗)
            features.get('depth', 0) / 10.0,      # 정규화 (최대 깊이 10)
            features.get('gate_count', 0) / 200.0, # 정규화 (최대 게이트 200개)
            features.get('cnot_count', 0) / 100.0, # 정규화
            features.get('single_qubit_gates', 0) / 150.0, # 정규화
            features.get('unique_gate_types', 0) / 8.0,    # 정규화
            features.get('cnot_connections', 0) / 50.0,    # 정규화
        ])
        
        # 커플링 특성
        coupling_features = torch.FloatTensor([
            features.get('coupling_density', 0.0),
            features.get('max_degree', 0) / 10.0,  # 정규화
            features.get('avg_degree', 0.0) / 5.0, # 정규화
            features.get('connectivity_ratio', 0.0),
            features.get('diameter', 0) / 20.0,    # 정규화
            features.get('clustering_coefficient', 0.0),
        ])
        
        # 파라미터 특성
        param_features = torch.FloatTensor([
            features.get('param_count', 0) / 50.0,  # 정규화
            features.get('param_mean', 0.0) / (2 * np.pi),  # 정규화
            features.get('param_std', 0.0) / np.pi,         # 정규화
            features.get('param_min', 0.0) / (2 * np.pi),   # 정규화
            features.get('param_max', 0.0) / (2 * np.pi),   # 정규화
        ])
        
        # 측정 통계 특성
        measurement_features = torch.FloatTensor([
            features.get('entropy', 0.0) / 20.0,  # 정규화 (최대 엔트로피 ~20)
            features.get('zero_state_probability', 0.0),
            features.get('concentration', 0.0),
            features.get('measured_states', 0) / 1000.0,  # 정규화
            features.get('top_1_probability', 0.0),
            features.get('top_5_probability', 0.0),
            features.get('top_10_probability', 0.0),
        ])
        
        # 시퀀스 특성
        sequence_features = torch.FloatTensor([
            features.get('sequence_length', 0) / MAX_SEQUENCE_LENGTH,
            features.get('unique_gates_in_sequence', 0) / 8.0,
            features.get('param_gate_ratio', 0.0),
            features.get('two_qubit_gate_ratio', 0.0),
        ])
        
        # 하드웨어 특성
        hardware_features = torch.FloatTensor([
            features.get('gate_overhead', 1.0),
            features.get('depth_overhead', 1.0),
            features.get('transpiled_depth', 0) / 50.0,  # 정규화
            features.get('transpiled_gate_count', 0) / 500.0,  # 정규화
        ])
        
        # 모든 특성 결합
        combined_features = torch.cat([
            structural_features,
            coupling_features, 
            param_features,
            measurement_features,
            sequence_features,
            hardware_features
        ])
        
        # 타겟 값들 (예측할 값들)
        targets = torch.FloatTensor([
            features.get('fidelity', 0.0),
            features.get('normalized_expressibility', 0.0),
            features.get('expressibility_distance', 0.0) / 1e-3,  # 정규화
        ])
        
        return {
            'gate_sequence': gate_sequence,
            'qubit_sequence': qubit_sequence, 
            'param_sequence': param_sequence,
            'gate_type_sequence': gate_type_sequence,
            'features': combined_features,
            'targets': targets,
            'circuit_id': circuit.get('circuit_id', 'unknown')
        }

#################################################
# 2. 우리 데이터에 맞는 트랜스포머 모델
#################################################

class QuantumCircuitTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=9,  # 게이트 타입 수 (0~8)
        max_qubits=127,
        sequence_length=MAX_SEQUENCE_LENGTH,
        d_model=LATENT_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        feature_dim=33,  # 결합된 특성 차원
        output_dim=3,    # 피델리티, 정규화된 표현력, 표현력 거리
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # 임베딩 레이어들
        self.gate_embedding = nn.Embedding(vocab_size, d_model // 4)
        self.qubit_embedding = nn.Embedding(max_qubits + 2, d_model // 4)  # +2 for padding (-1) and special tokens
        self.gate_type_embedding = nn.Embedding(3, d_model // 4)  # 0, 1, 2
        self.param_projection = nn.Linear(1, d_model // 4)
        
        # 시퀀스 결합 프로젝션
        self.sequence_projection = nn.Linear(d_model, d_model)
        
        # 포지셔널 인코딩
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_model))
        
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 특성 처리 네트워크
        self.feature_network = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 최종 예측 네트워크
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 시퀀스 + 특성
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, gate_seq, qubit_seq, param_seq, gate_type_seq, features):
        batch_size = gate_seq.size(0)
        
        # 큐빗 시퀀스 처리 (음수 값을 특별한 토큰으로 변환)
        qubit_seq_processed = qubit_seq.clone()
        qubit_seq_processed[qubit_seq_processed < 0] = 127 + 1  # 패딩 토큰
        
        # 각 컴포넌트 임베딩
        gate_emb = self.gate_embedding(gate_seq)  # [B, L, d_model//4]
        qubit_emb = self.qubit_embedding(qubit_seq_processed)  # [B, L, d_model//4]
        gate_type_emb = self.gate_type_embedding(gate_type_seq)  # [B, L, d_model//4]
        param_emb = self.param_projection(param_seq.unsqueeze(-1))  # [B, L, d_model//4]
        
        # 시퀀스 임베딩 결합
        sequence_emb = torch.cat([gate_emb, qubit_emb, gate_type_emb, param_emb], dim=-1)  # [B, L, d_model]
        sequence_emb = self.sequence_projection(sequence_emb)
        
        # 포지셔널 인코딩 추가
        sequence_emb = sequence_emb + self.pos_encoding
        sequence_emb = self.dropout(sequence_emb)
        
        # 패딩 마스크 생성 (게이트 시퀀스가 0인 위치)
        padding_mask = (gate_seq == 0)
        
        # 트랜스포머 인코더 통과
        transformer_output = self.transformer(sequence_emb, src_key_padding_mask=padding_mask)
        
        # 시퀀스 표현 (평균 풀링, 패딩 제외)
        mask = (~padding_mask).float().unsqueeze(-1)
        sequence_repr = (transformer_output * mask).sum(dim=1) / mask.sum(dim=1)
        
        # 특성 처리
        feature_repr = self.feature_network(features)
        
        # 시퀀스와 특성 결합
        combined_repr = torch.cat([sequence_repr, feature_repr], dim=-1)
        
        # 최종 예측
        predictions = self.prediction_head(combined_repr)
        
        return predictions

#################################################
# 3. 훈련 및 평가 함수들
#################################################

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """모델 훈련"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            
            # 데이터 GPU로 이동
            gate_seq = batch['gate_sequence'].to(DEVICE)
            qubit_seq = batch['qubit_sequence'].to(DEVICE)
            param_seq = batch['param_sequence'].to(DEVICE)
            gate_type_seq = batch['gate_type_sequence'].to(DEVICE)
            features = batch['features'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            # 예측
            predictions = model(gate_seq, qubit_seq, param_seq, gate_type_seq, features)
            
            # 손실 계산
            loss = criterion(predictions, targets)
            
            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                gate_seq = batch['gate_sequence'].to(DEVICE)
                qubit_seq = batch['qubit_sequence'].to(DEVICE)
                param_seq = batch['param_sequence'].to(DEVICE)
                gate_type_seq = batch['gate_type_sequence'].to(DEVICE)
                features = batch['features'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                predictions = model(gate_seq, qubit_seq, param_seq, gate_type_seq, features)
                loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        # 평균 손실 계산
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 학습률 스케줄링
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 최고 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_quantum_transformer.pth')
            print(f"  ✅ 새로운 최고 모델 저장! (Val Loss: {best_val_loss:.6f})")
        
        print()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """모델 평가"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_circuit_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="평가 중"):
            gate_seq = batch['gate_sequence'].to(DEVICE)
            qubit_seq = batch['qubit_sequence'].to(DEVICE)
            param_seq = batch['param_sequence'].to(DEVICE)
            gate_type_seq = batch['gate_type_sequence'].to(DEVICE)
            features = batch['features'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            predictions = model(gate_seq, qubit_seq, param_seq, gate_type_seq, features)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_circuit_ids.extend(batch['circuit_id'])
    
    # 결과 결합
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # 메트릭 계산
    target_names = ['Fidelity', 'Normalized Expressibility', 'Expressibility Distance']
    
    print("\n📊 평가 결과:")
    print("=" * 60)
    
    for i, name in enumerate(target_names):
        mse = mean_squared_error(targets[:, i], predictions[:, i])
        r2 = r2_score(targets[:, i], predictions[:, i])
        
        print(f"{name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  R²: {r2:.4f}")
        print()
    
    return predictions, targets, all_circuit_ids

def plot_training_curves(train_losses, val_losses):
    """훈련 곡선 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(predictions, targets, target_names):
    """예측 결과 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, ax) in enumerate(zip(target_names, axes)):
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.6)
        ax.plot([targets[:, i].min(), targets[:, i].max()], 
                [targets[:, i].min(), targets[:, i].max()], 'r--', lw=2)
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} Prediction')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

#################################################
# 4. 메인 실행 함수
#################################################

def main():
    """메인 실행 함수"""
    print("🚀 양자 회로 트랜스포머 모델 훈련 시작!")
    print(f"디바이스: {DEVICE}")
    
    # 데이터 로드
    data_dir = "grid_ansatz/grid_circuits/training_data"
    
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        print("먼저 mega job을 실행하여 데이터를 생성하세요.")
        return
    
    # 데이터셋 생성
    print("\n📂 데이터셋 로드 중...")
    train_dataset = QuantumMegaJobDataset(data_dir, split='train')
    val_dataset = QuantumMegaJobDataset(data_dir, split='val')
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 모델 생성
    print("\n🤖 모델 초기화 중...")
    model = QuantumCircuitTransformer().to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터 수: {total_params:,}")
    print(f"훈련 가능한 파라미터 수: {trainable_params:,}")
    
    # 모델 훈련
    print("\n🏋️ 모델 훈련 시작...")
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    # 훈련 곡선 시각화
    plot_training_curves(train_losses, val_losses)
    
    # 최고 모델 로드
    print("\n📥 최고 모델 로드 중...")
    model.load_state_dict(torch.load('best_quantum_transformer.pth'))
    
    # 평가
    print("\n📊 모델 평가 중...")
    predictions, targets, circuit_ids = evaluate_model(model, val_loader)
    
    # 예측 결과 시각화
    target_names = ['Fidelity', 'Normalized Expressibility', 'Expressibility Distance']
    plot_predictions(predictions, targets, target_names)
    
    # 결과 저장
    results_df = pd.DataFrame({
        'circuit_id': circuit_ids,
        'true_fidelity': targets[:, 0],
        'pred_fidelity': predictions[:, 0],
        'true_norm_expr': targets[:, 1],
        'pred_norm_expr': predictions[:, 1],
        'true_expr_dist': targets[:, 2],
        'pred_expr_dist': predictions[:, 2]
    })
    
    results_df.to_csv('prediction_results.csv', index=False)
    print("\n💾 결과가 'prediction_results.csv'에 저장되었습니다.")
    
    print("\n🎉 훈련 완료!")

if __name__ == "__main__":
    main() 