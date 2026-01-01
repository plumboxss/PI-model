# 아키텍처 검증 보고서 (Architectural Audit)

## 검증 기준: 프로젝트 목표
- **목표**: preference-conditioned reward model + few-shot adaptation
- **핵심**: 소수의 비교(context K개)로 z 추정, query에 일반화
- **y 정의**: trajectory pair (s1 vs s2) 당 1개 preference label

---

## A) 구조 정합성 검증

### A1. 시간축 T와 비교축 K 완전 분리 여부

**검증 위치**: `src/models/vae.py::encode_context()`, `src/models/trajectory_encoder.py`, `src/models/set_encoder.py`

**결과**: ✅ **OK**

**이유**:
- `TrajectoryEncoder.forward()`: 입력 `(B*K, T, D_sa)` → 출력 `(B*K, d)`. T는 여기서만 처리되고 mean pooling으로 고정 길이로 변환됨.
- `SetEncoder.forward()`: 입력 `(B, K, h_dim)` → 출력 `(B, latent_dim)`. K는 여기서만 처리되고, T 정보는 이미 trajectory encoder에서 제거됨.
- `encode_context()`에서 `(B, K, T, D_sa)` → `(B*K, T, D_sa)` → `(B*K, d)` → `(B, K, h_dim)` → `(B, z_dim)` 순서로 처리되어 T와 K가 명확히 분리됨.

---

### A2. Set encoder에 timestep 정보 유입 경로 확인

**검증 위치**: `src/models/set_encoder.py::forward()`, `src/models/vae.py::encode_context()`

**결과**: ✅ **OK**

**이유**:
- Set encoder 입력: `H: (B, K, h_dim)` - 이미 trajectory embedding으로 변환된 고정 길이 벡터
- Trajectory encoder에서 `mean(dim=1)`로 T 차원을 제거하므로, Set encoder에는 T 정보가 전혀 유입되지 않음.
- 코드 경로 추적: `context_s1 (B,K,T,D)` → `s1_flat (B*K,T,D)` → `trajectory_encoder` → `e1 (B*K,d)` → `pair_encoder` → `h (B*K,h_dim)` → `H (B,K,h_dim)` → `set_encoder`. T는 trajectory encoder 단계에서 완전히 제거됨.

---

### A3. y를 timestep으로 expand/broadcast하는 코드 제거 여부

**검증 위치**: `src/models/vae.py`, `src/models/pair_encoder.py`, 전체 코드베이스 grep

**결과**: ✅ **OK**

**이유**:
- `PairEncoder.forward()`: `y: (B, 1)` 그대로 사용, expand 없음.
- `encode_context()`: `y_flat = context_y.view(B * K, 1)` - reshape만 하고 expand 없음.
- `grep` 결과: `reward_decoder.py`에서 z만 expand하고, y 관련 expand/repeat 코드 없음.
- 이전 코드의 `y.reshape(batch_size, 1, 1).expand(batch_size, seq_len, 1)` 패턴 완전 제거됨.

---

## B) 데이터 누수 / 메타러닝 정합성 검증

### B1. Context와 Query가 같은 pair를 재사용하지 않는지

**검증 위치**: `src/data/loader.py::ContextQueryDataset.__getitem__()`

**결과**: ✅ **OK**

**이유**:
- `__init__()`에서 `context_candidates = [i for i in range(len(indices)) if i != query_idx]`로 query_idx를 제외.
- `context_indices = np.random.choice(context_candidates, ...)`로 query와 겹치지 않도록 샘플링.
- `__getitem__()`에서 `real_idx = group_indices[ctx_idx]`와 `query_real_idx = group_indices[query_idx]`로 다른 인덱스 사용.

---

### B2. 같은 trajectory가 context와 query에 동시에 들어가는지

**검증 위치**: `src/data/loader.py::ContextQueryDataset.__getitem__()`, `scripts/build_preference_dataset.py`

**결과**: ⚠️ **부분적 문제 발견**

**문제점**:
- `build_preference_dataset.py`에서 각 pair는 `(traj_idx1, traj_idx2, label)`로 저장됨.
- `ContextQueryDataset`은 **pair 인덱스** 기준으로 context/query를 분리함.
- 따라서 **같은 trajectory가 다른 pair에서 context와 query에 동시에 나타날 수 있음**.

**예시**:
- Pair 0: (traj_A, traj_B, y=1) → context로 사용
- Pair 1: (traj_A, traj_C, y=0) → query로 사용
- 결과: traj_A가 context와 query에 동시에 등장

**영향**:
- 메타러닝 관점에서 약한 데이터 누수. 완전한 few-shot 시나리오는 아니지만, 실용적으로는 문제가 될 수 있음.

**최소 수정**:
- `ContextQueryDataset.__init__()`에서 trajectory 레벨로 그룹핑하여, 같은 trajectory가 context와 query에 동시에 나타나지 않도록 수정 필요.

---

### B3. model_id가 의미 있는 그룹인지

**검증 위치**: `scripts/build_preference_dataset.py::main()`, `src/data/loader.py::get_datasets()`

**결과**: ⚠️ **의미는 있으나 제한적**

**현재 구현**:
- `build_preference_dataset.py:120`: `model_id = cluster_id` (K-Means 클러스터 ID)
- 클러스터는 **특징 기반**으로 생성됨 (jerk, pitch, settling_time)
- 같은 클러스터 = 비슷한 특징을 가진 궤적들

**문제점**:
- `model_id`는 **궤적의 특징**을 나타내지, **실제 사용자/오라클의 선호 정체성**을 나타내지 않음.
- 프로젝트 목표는 "같은 annotator/user의 선호"를 학습하는 것인데, 현재는 "비슷한 특징을 가진 궤적 그룹"으로 묶임.
- **같은 사용자가 다른 클러스터의 궤적을 선호할 수도 있음** (예: 사용자는 항상 같은 선호 패턴을 가지지만, 궤적은 다양한 특징을 가질 수 있음).

**영향**:
- Few-shot adaptation 목표와 약간 어긋남. 실제로는 "특징 기반 그룹"에 적응하는 것이지, "사용자 선호"에 적응하는 것이 아님.

**최소 수정**:
- `build_preference_dataset.py`에서 실제 사용자/오라클 ID를 생성하거나, 궤적 생성 시점에 사용자 ID를 부여해야 함.
- 또는 현재 구조를 유지하되, "특징 기반 선호 그룹에 적응"이라는 목표로 명확히 문서화.

---

## C) 학습 의미 검증

### C1. K=1일 때 forward_legacy 의미

**검증 위치**: `src/models/vae.py::forward_legacy()`

**결과**: ⚠️ **의미적으로 타당하나 제한적**

**현재 구현**:
```python
context_s1 = s1.unsqueeze(1)  # (B, 1, T, D_sa) - K=1
query_s1 = s1  # 같은 trajectory를 query로도 사용
```

**문제점**:
- **Context와 Query가 동일한 pair를 사용함** → 데이터 누수
- K=1이므로 z 추정에 정보가 부족함
- 실제 few-shot adaptation 시나리오와 다름

**의미**:
- 단순히 "기존 코드 호환성"을 위한 편의 함수
- 실제 few-shot 학습에는 부적합

**권장사항**:
- `forward_legacy`는 하위 호환용으로만 사용하고, 실제 학습에는 context-query 구조 사용 권장.

---

### C2. z가 context에 따라 달라질 수 있는 구조인지

**검증 위치**: `src/models/vae.py::encode_context()`, `src/models/set_encoder.py`

**결과**: ✅ **OK**

**이유**:
- `encode_context()`: K개의 비교를 모두 사용하여 `(B, K, h_dim)` → `(B, latent_dim)` 변환
- `SetEncoder`: K에 대해 mean pooling 또는 attention으로 집계
- **K가 달라지면 입력 H가 달라지고, 따라서 z도 달라짐**
- 구조적으로 z는 context에 의존적임

**검증 코드**:
```python
# K=1일 때와 K=5일 때 z가 다름을 확인 가능
context_K1 = ...  # (B, 1, T, D)
context_K5 = ...  # (B, 5, T, D)
z1 = model.encode_context(...)[0]  # K=1
z5 = model.encode_context(...)[0]  # K=5
# z1 != z5 (일반적으로)
```

---

### C3. KL collapse 위험

**검증 위치**: `src/models/vae.py::forward()`, KL loss 계산

**결과**: ⚠️ **잠재적 위험 존재**

**현재 구현**:
- `learned_prior=False`: 표준 KL `KL(q(z|context) || N(0, I))` ✅ 안전
- `learned_prior=True`: `KL(q(z|context) || p_learned(z))` ⚠️ 위험

**문제점**:
- `learned_prior=True`일 때, prior가 학습 가능한 파라미터
- 만약 prior가 posterior와 너무 가까워지면 KL이 0에 수렴할 수 있음
- 하지만 현재 구현에서는 `prior_mean`, `prior_log_var`가 초기값 0으로 고정되어 있고, `requires_grad=learned_prior`로 제어됨.

**추가 확인 필요**:
- `learned_prior=True`일 때 실제로 prior가 학습되는지 확인 필요
- 현재 코드에서는 `learned_prior`가 True여도 prior 파라미터가 업데이트되는 경로가 명시적으로 보이지 않음 (optimizer에 포함되어야 함)

**최소 수정**:
- `learned_prior=True`일 때 prior 파라미터가 optimizer에 포함되는지 확인 필요.

---

## D) Reward / Loss 축 검증

### D1. Reward sum이 timestep 축(T)에 대해 정확히 수행되는가

**검증 위치**: `src/models/vae.py::forward()`, `src/models/reward_decoder.py`

**결과**: ✅ **OK**

**이유**:
- `decode_reward()`: `(B, T, 1)` 출력
- `forward()`: `R1 = r1.sum(dim=1) / self.scaling` - **dim=1이 T 축**
- `r1.shape = (B, T, 1)` → `sum(dim=1)` → `(B, 1)` ✅ 정확함

---

### D2. BT 확률 계산에 scaling/temperature 일관성

**검증 위치**: `src/models/vae.py::forward()`

**결과**: ✅ **OK**

**이유**:
- `R1 = r1.sum(dim=1) / self.scaling`
- `R2 = r2.sum(dim=1) / self.scaling`
- `p_hat = torch.sigmoid(R1 - R2)` - scaling이 일관되게 적용됨
- Temperature는 사용하지 않음 (sigmoid만 사용)

---

### D3. query_y shape/dtype이 BCE에 정확히 들어가는가

**검증 위치**: `src/models/vae.py::forward()`

**결과**: ✅ **OK**

**이유**:
- `query_y`: `(B, 1)` (DataLoader에서 `torch.tensor([query_label]).float()`)
- `p_hat`: `(B, 1)`
- `F.binary_cross_entropy(p_hat.view(-1, 1), query_y.view(-1, 1))` - 둘 다 `(B, 1)`로 변환되어 일치함

---

## E) 목표 적합성 최종 판단

### E1. Few-shot preference adaptation 구조적 가능성

**결과**: ✅ **구조적으로 가능함**

**이유**:
- Context-Query 구조가 명확히 구현됨
- z는 context K개에만 의존
- Loss는 query에만 계산
- 메타러닝 프레임워크와 일치

**제한사항**:
- B2에서 지적한 trajectory 레벨 데이터 누수 (pair 레벨은 분리되지만 trajectory는 재사용 가능)
- B3에서 지적한 model_id의 의미 제한 (특징 기반 그룹, 실제 사용자 ID 아님)

---

### E2. 개념적 결함 vs 구현 디테일

**결과**: **구현 디테일 문제 2개 발견**

#### 문제 1: Trajectory 레벨 데이터 누수 (구현 디테일)
- **위치**: `src/data/loader.py::ContextQueryDataset`
- **문제**: 같은 trajectory가 context와 query에 동시에 나타날 수 있음
- **영향**: 약한 데이터 누수, 완전한 few-shot은 아님
- **수정**: Trajectory ID 기준으로 그룹핑하여 분리

#### 문제 2: model_id 의미 제한 (개념적 제한)
- **위치**: `scripts/build_preference_dataset.py`
- **문제**: model_id가 실제 사용자 ID가 아니라 특징 기반 클러스터 ID
- **영향**: "사용자 선호 적응"이 아니라 "특징 그룹 적응"
- **수정**: 실제 사용자/오라클 ID를 생성하거나, 목표를 명확히 문서화

---

## 최종 판정

### ✅ 구조적 정합성: **우수**
- T와 K 완전 분리 ✅
- y expand 제거 ✅
- 아키텍처 명확 ✅

### ⚠️ 메타러닝 정합성: **부분적 문제**
- Pair 레벨 분리: ✅ OK
- Trajectory 레벨 분리: ⚠️ 문제 (같은 trajectory 재사용 가능)
- model_id 의미: ⚠️ 제한적 (특징 기반, 사용자 ID 아님)

### ✅ 학습 의미: **타당**
- z는 context에 의존적 ✅
- KL collapse: ⚠️ learned_prior 옵션 확인 필요

### ✅ Reward/Loss: **정확**
- 모든 축 계산 정확 ✅

### 🎯 목표 적합성: **구조적으로 가능, 구현 디테일 개선 필요**

**결론**: 리팩토링은 **구조적으로 올바르고 목표에 부합**하나, **2개의 구현 디테일 문제**가 있어 완전한 few-shot adaptation을 위해서는 수정이 필요함.

---

## 권장 수정사항

### 우선순위 1: Trajectory 레벨 데이터 누수 수정
**파일**: `src/data/loader.py::ContextQueryDataset`
**수정**: Trajectory ID를 추적하여 같은 trajectory가 context와 query에 동시에 나타나지 않도록 수정

### 우선순위 2: model_id 의미 명확화
**옵션 A**: 실제 사용자/오라클 ID 생성 로직 추가
**옵션 B**: 현재 구조를 "특징 기반 그룹 적응"으로 명확히 문서화

### 우선순위 3: learned_prior 옵션 검증
**파일**: `scripts/train_model.py`, `src/models/vae.py`
**확인**: ✅ **OK** - `reward_model.parameters()`에 prior 파라미터가 포함됨 (VAE 모델의 일부이므로 자동 포함)

