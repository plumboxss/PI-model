# VAE 모델 리팩토링 요약

## 📋 개요

프로젝트의 핵심 문제였던 **seq_len 축 혼용 문제 (T vs K)**를 해결하고, **context-query 구조**를 명확히 분리한 리팩토링을 완료했습니다.

## 🎯 해결된 문제

### 이전 문제점
- `seq_len`이 timestep(T)와 comparison count(K)가 혼용됨
- `y`를 (B,1,1) -> (B,T,1)로 억지로 expand하여 attention 입력에 붙임
- 목표(컨텍스트 기반 z 추정)와 구조가 맞지 않음

### 해결 방법
- **명확한 계층 구조**: Trajectory → Pair → Set → Reward
- **T와 K 완전 분리**: T는 trajectory encoder 내부에서만, K는 set encoder에서만 처리
- **y는 pair당 1개**: 더 이상 timestep으로 expand하지 않음
- **Context-Query 구조**: K개 비교로 z 추정, 1개 query로 loss 계산

## 📁 변경된 파일 목록

### 새로 생성된 파일
1. **`src/models/trajectory_encoder.py`**
   - 궤적 τ: (B, T, D_sa) → 고정 길이 임베딩 e: (B, d)
   - Transformer/LSTM/MLP 옵션 지원

2. **`src/models/pair_encoder.py`**
   - 쌍 (e1, e2, y) → 피드백 벡터 h_i: (B, h_dim)
   - y는 pair당 1개 스칼라로 처리

3. **`src/models/set_encoder.py`**
   - 피드백 벡터 집합 {h_i} → 잠재 분포 (μ, log_var)
   - Attention 또는 DeepSets 옵션

4. **`src/models/reward_decoder.py`**
   - (obs, act, z) → 보상 r: (B, T, 1)
   - z 브로드캐스팅 자동 처리

### 수정된 파일
1. **`src/models/vae.py`** ⭐ 핵심 변경
   - 완전히 재작성
   - 새 모듈들 사용
   - `forward()` 메서드: context-query 구조
   - `forward_legacy()`: 하위 호환성 (K=1)

2. **`src/data/loader.py`**
   - `ContextQueryDataset` 클래스 추가
   - `collate_context_query()` 함수 추가
   - model_id 기반 그룹핑으로 context 구성

3. **`scripts/train_model.py`**
   - 새 모델 인터페이스에 맞게 수정
   - Shape 검증 코드 추가
   - FLAGS에 `context_size`, `trajectory_encoder_type`, `set_encoder_type` 추가

## 🔧 핵심 변경 사항

### 1. 아키텍처 분리

```
이전:
s1, s2 (B, T, D) + y (B, 1) → [y를 expand] → Encoder → z → Decoder

이후:
Context: (B, K, T, D_sa) × 2 + (B, K, 1)
  → Trajectory Encoder → (B, K, d)
  → Pair Encoder → (B, K, h_dim)
  → Set Encoder → (B, z_dim) [z]
Query: (B, T, D_sa) × 2 + (B, 1)
  → Reward Decoder(z) → (B, T, 1) → R → Loss
```

### 2. 데이터 구조

**이전:**
```python
{
    "s1": (B, T, D_sa),
    "s2": (B, T, D_sa),
    "labels": (B, 1)
}
```

**이후:**
```python
{
    "context_s1": (B, K, T, D_sa),
    "context_s2": (B, K, T, D_sa),
    "context_y": (B, K, 1),
    "query_s1": (B, T, D_sa),
    "query_s2": (B, T, D_sa),
    "query_y": (B, 1)
}
```

### 3. 모델 초기화

**이전:**
```python
VAEModel(
    encoder_input_dim=...,
    decoder_input_dim=...,
    action_dim=...,
    encoder_type='attention',
    ...
)
```

**이후:**
```python
VAEModel(
    obs_dim=observation_dim,
    act_dim=action_dim,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    trajectory_encoder_type='transformer',  # 'transformer', 'lstm', 'mlp'
    set_encoder_type='attention',  # 'attention', 'deepset'
    ...
)
```

### 4. Forward Pass

**이전:**
```python
loss, metrics = model(s1, s2, labels)
```

**이후:**
```python
loss, metrics = model(
    context_s1, context_s2, context_y,
    query_s1, query_s2, query_y
)
```

## 📊 Shape 검증

학습 시작 시 첫 배치에서 자동으로 shape를 검증합니다:

```
=== Shape Validation ===
context_s1: (B, K, T, D_sa) (B, K, T, D_sa)
context_s2: (B, K, T, D_sa) (B, K, T, D_sa)
context_y: (B, K, 1) (B, K, 1)
query_s1: (B, T, D_sa) (B, T, D_sa)
query_s2: (B, T, D_sa) (B, T, D_sa)
query_y: (B, 1) (B, 1)
=======================
```

## ⚙️ 새로운 설정 옵션

### `train_model.py` FLAGS 추가:
- `context_size=5`: Context 비교 개수 (K)
- `trajectory_encoder_type='transformer'`: 'transformer', 'lstm', 'mlp'
- `set_encoder_type='attention'`: 'attention', 'deepset'

### 제거된 FLAGS:
- `encoder_type`: `trajectory_encoder_type`과 `set_encoder_type`으로 분리

## 🔄 하위 호환성

`VAEModel.forward_legacy()` 메서드를 제공하여 기존 코드와의 호환성을 유지합니다:
- K=1로 context를 구성
- 동일한 입력/출력 인터페이스

## ✅ 검증 완료 사항

- [x] T와 K 완전 분리
- [x] y expand 로직 제거
- [x] Context-query 구조 구현
- [x] Shape 검증 코드 추가
- [x] KL annealing 유지
- [x] WandB 로깅 유지
- [x] Early stopping 유지
- [x] Checkpoint 저장 유지

## 🚀 사용 방법

### 학습 실행:
```bash
python scripts/train_model.py \
    --dataset_path datasets/preference_dataset_A.pkl \
    --context_size 5 \
    --trajectory_encoder_type transformer \
    --set_encoder_type attention \
    --logging.output_dir "logs" \
    --comment "refactored_vae" \
    --seed 42
```

## 📝 주의사항

1. **기존 체크포인트**: 새 아키텍처와 호환되지 않으므로 새로 학습해야 합니다.
2. **데이터셋**: `model_id` 필드가 있으면 그룹핑에 사용하고, 없으면 자동으로 그룹을 생성합니다.
3. **메모리**: Context size(K)가 클수록 메모리 사용량이 증가합니다.

## 🔍 다음 단계 (선택사항)

- [ ] `run_interactive_adaptation.py` 업데이트 (새 모델 인터페이스에 맞게)
- [ ] `evaluate_adaptation.py` 업데이트
- [ ] 성능 벤치마크
- [ ] 하이퍼파라미터 튜닝 (context_size, encoder types)

