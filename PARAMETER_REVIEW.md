# 파라미터 점검 보고서

## 📊 데이터 생성 파라미터 검토

### 제안된 설정
- **궤적 수**: 500개
- **클러스터 수**: 16개
- **선호도 쌍 수**: 20,000개

### 분석

#### ✅ 궤적 수 (500개)
- **평균 궤적/클러스터**: 500 / 16 = **약 31개**
- **평가**: ✅ **적절함**
  - 각 클러스터당 최소 6개 이상 필요 (context_size=5 + query=1)
  - 31개는 충분한 여유가 있음
  - 클러스터링 품질을 위해 최소 20-30개 이상 권장

#### ✅ 클러스터 수 (16개)
- **평가**: ✅ **적절함**
  - `latent_dim=8`로 충분히 표현 가능
  - 다양한 선호도 패턴 학습 가능
  - 과도한 클러스터링 방지 (over-clustering)

#### ⚠️ 선호도 쌍 수 (20,000개)
- **평균 쌍/클러스터**: 20,000 / 16 = **약 1,250개**
- **평가**: ✅ **충분함**
  - 각 클러스터당 충분한 학습 데이터
  - Train/Test split 후에도 여유 있음
  - Context-query 구조를 고려해도 충분

### 데이터 생성 파라미터 최종 평가: ✅ **적절함**

---

## 🎓 학습 파라미터 점검

### 현재 설정

#### 1. 모델 아키텍처 파라미터

| 파라미터 | 현재 값 | 평가 | 권장 사항 |
|---------|---------|------|----------|
| `latent_dim` | **8** | ✅ 적절 | 16개 클러스터 표현에 충분 |
| `hidden_dim` | 256 | ✅ 적절 | 모델 용량 적절 |
| `trajectory_encoder_type` | 'transformer' | ✅ 적절 | 시간 정보 활용 |
| `set_encoder_type` | 'attention' | ✅ 적절 | 비교 간 상호작용 모델링 |
| `n_heads` | 4 | ✅ 적절 | hidden_dim=256에 적합 |
| `n_layers` | 2 | ✅ 적절 | 과적합 방지 |

#### 2. 학습 하이퍼파라미터

| 파라미터 | 현재 값 | 평가 | 권장 사항 |
|---------|---------|------|----------|
| `batch_size` | 256 | ⚠️ **재검토 필요** | 데이터 크기 고려 시 조정 가능 |
| `lr` | 1e-3 | ✅ 적절 | Adam optimizer에 적합 |
| `n_epochs` | 500 | ✅ 적절 | 충분한 학습 시간 |
| `context_size` | 5 | ✅ 적절 | Few-shot adaptation에 적합 |

#### 3. VAE 특화 파라미터

| 파라미터 | 현재 값 | 평가 | 권장 사항 |
|---------|---------|------|----------|
| `kl_weight` | 1.0 | ⚠️ **재검토 필요** | Annealing 사용 시 초기값으로 적절 |
| `use_annealing` | True | ✅ 적절 | KL collapse 방지 |
| `annealer_type` | 'cosine' | ✅ 적절 | 부드러운 증가 |
| `annealer_cycles` | 4 | ✅ 적절 | 500 epochs에 적합 |
| `annealer_baseline` | 0.0 | ✅ 적절 | 0에서 시작 |
| `learned_prior` | False | ✅ 적절 | 표준 N(0,I) prior |
| `reward_scaling` | 1.0 | ✅ 적절 | 기본값 |

#### 4. 기타 파라미터

| 파라미터 | 현재 값 | 평가 | 권장 사항 |
|---------|---------|------|----------|
| `eval_freq` | 50 | ✅ 적절 | 500 epochs에 적합 |
| `save_freq` | 50 | ✅ 적절 | 충분한 체크포인트 |
| `early_stop` | False | ✅ 적절 | 수동 모니터링 권장 |
| `seed` | 42 | ✅ 적절 | 재현성 보장 |

---

## 🔍 상세 분석

### 1. Batch Size (256)

**현재 설정**: `batch_size=256`

**데이터 크기 고려**:
- 총 20,000개 쌍
- Train/Test split (80/20): Train 16,000개, Test 4,000개
- 배치 수: 16,000 / 256 = **약 62.5 배치/epoch**

**평가**:
- ✅ **적절함**: 배치 수가 충분함
- ⚠️ **고려사항**: GPU 메모리에 따라 조정 가능
  - Context size=5, T=500, D_sa=11일 때 메모리 사용량 계산 필요
  - 메모리 부족 시 128 또는 64로 감소 고려

**권장**: 현재 값 유지, 메모리 문제 시 128로 감소

---

### 2. Learning Rate (1e-3)

**현재 설정**: `lr=1e-3`

**평가**:
- ✅ **적절함**: Adam optimizer의 표준 학습률
- Transformer 기반 모델에 적합
- VAE 학습에 일반적으로 사용되는 값

**권장**: 현재 값 유지

---

### 3. KL Weight & Annealing

**현재 설정**:
- `kl_weight=1.0`
- `use_annealing=True`
- `annealer_type='cosine'`
- `annealer_cycles=4`

**Annealing 계산**:
- Total steps per cycle: 500 / 4 = 125 epochs
- Cosine annealing: 0 → 1.0으로 부드럽게 증가

**평가**:
- ✅ **적절함**: KL collapse 방지에 효과적
- ⚠️ **모니터링 필요**: KL loss가 너무 작거나 크지 않은지 확인

**권장**: 현재 값 유지, 학습 곡선 모니터링

---

### 4. Context Size (5)

**현재 설정**: `context_size=5`

**평가**:
- ✅ **적절함**: Few-shot adaptation 목표에 부합
- 각 그룹당 최소 6개 pair 필요 (5 context + 1 query)
- 500개 궤적, 16개 클러스터: 평균 31개/클러스터 → 충분

**권장**: 현재 값 유지

---

### 5. Latent Dimension (8)

**현재 설정**: `latent_dim=8` (방금 변경)

**평가**:
- ✅ **적절함**: 16개 클러스터 표현에 충분
- 정보 압축과 표현력의 균형
- 과적합 방지

**권장**: 현재 값 유지

---

## 📈 데이터-모델 파라미터 매칭 검증

### 클러스터 수 vs Latent Dimension

- **클러스터**: 16개
- **Latent dim**: 8차원
- **평가**: ✅ **적절함**
  - 8차원으로 16개 클러스터 표현 가능
  - 각 클러스터가 고유한 z 영역을 차지할 수 있음

### 데이터 크기 vs Batch Size

- **Train samples**: 약 16,000개 (80% split)
- **Batch size**: 256
- **Batches/epoch**: 약 62.5
- **평가**: ✅ **적절함**

### Context Size vs 데이터 분포

- **Context size**: 5
- **평균 pair/클러스터**: 약 1,250개
- **최소 필요 pair**: 6개 (5 context + 1 query)
- **평가**: ✅ **충분함**

---

## ⚠️ 주의사항 및 권장사항

### 1. 메모리 사용량

**예상 메모리**:
- Context: (B=256, K=5, T=500, D=11) = 약 5.6M elements
- Query: (B=256, T=500, D=11) = 약 1.4M elements
- **총 예상**: GPU 메모리 2-4GB (모델 포함)

**권장**: 메모리 부족 시 `batch_size`를 128로 감소

### 2. 학습 모니터링

**중요 지표**:
- Reconstruction loss (BCE): 감소해야 함
- KL loss: 적절히 유지 (너무 작으면 collapse, 너무 크면 표현력 저하)
- Accuracy: 증가해야 함
- KL weight (annealing): 0 → 1.0으로 부드럽게 증가 확인

### 3. 조기 종료 고려

**현재**: `early_stop=False`

**권장**: 학습 곡선을 모니터링하고, 수동으로 조기 종료 결정
- 또는 `early_stop=True`, `patience=20-30` 설정 고려

---

## ✅ 최종 권장 설정

### 데이터 생성
```bash
python scripts/generate_data.py --num-episodes 500 --dataset-id A

python scripts/build_preference_dataset.py \
    --input_path artifacts/A/datasets/500.pkl \
    --output_path datasets/preference_dataset_A.pkl \
    --num_clusters 16 \
    --num_pairs 20000
```

### 모델 학습
```bash
python scripts/train_model.py \
    --dataset_path datasets/preference_dataset_A.pkl \
    --latent_dim 8 \
    --batch_size 256 \
    --lr 1e-3 \
    --n_epochs 500 \
    --context_size 5 \
    --use_annealing True \
    --annealer_type cosine \
    --annealer_cycles 4 \
    --kl_weight 1.0 \
    --hidden_dim 256 \
    --trajectory_encoder_type transformer \
    --set_encoder_type attention \
    --logging.output_dir "logs" \
    --comment "latent8_cluster16" \
    --seed 42
```

---

## 📊 파라미터 요약

### ✅ 적절한 파라미터
- 궤적 수: 500개
- 클러스터 수: 16개
- 선호도 쌍: 20,000개
- `latent_dim`: 8
- `hidden_dim`: 256
- `lr`: 1e-3
- `n_epochs`: 500
- `context_size`: 5
- `use_annealing`: True
- `annealer_type`: 'cosine'
- `annealer_cycles`: 4

### ⚠️ 모니터링 필요한 파라미터
- `batch_size`: 256 (메모리 문제 시 128로 감소)
- `kl_weight`: 1.0 (annealing과 함께 모니터링)

### 📝 결론

**전체 파라미터 설정**: ✅ **적절함**

모든 파라미터가 프로젝트 목표와 데이터 크기에 적합하게 설정되어 있습니다. 학습 중 메모리 문제가 발생하면 `batch_size`만 조정하면 됩니다.

