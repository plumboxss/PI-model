## 연구과정 정리 (Research Log)

### 0) 문제 진단: 왜 Posterior Collapse가 발생했나
관측/라벨 구조상 다음 조건이 만족되면 posterior collapse가 쉽게 발생합니다.
- 디코더가 \(z\) 없이도 \(y\)를 잘 맞출 수 있음(“지름길” 존재)
- 입력 스케일이 한쪽으로 치우쳐 작은 신호가 묻힘
- sigmoid가 포화되어 reconstruction gradient가 약해짐
- KL 압력이 상대적으로 강해져 \(q(z|c)\rightarrow p(z)\)로 수렴

### 1) 시뮬레이션 다양성 강화
- PD(+reference shaping) 제어기(`kp/kd/shaping_factor`) 도입
- 범프 높이/폭 랜덤화(`bump_height`, `bump_half_width`)
- 피처 확장: `rms_acceleration` 추가

### 2) Settling time 정의 변경(제어이론식)
기존 RMS 기반 규칙이 dt에 의해 양자화/고정되는 문제가 있어, 아래 정의로 변경했습니다.
- final_value: tail 구간 평균
- band: \(|x(t)-final|\le \max(rel\_tol|final|, abs\_tol)\)
- hold_time 동안 band 유지 시점 = settling time

### 3) 오라클/데이터셋 리팩토링
- K-Means 제거
- feature 기반 오라클로 라벨링 전환(선호 데이터 생성 파이프라인 확립)
- 유저별 쿼터를 균등 분배(각 유저가 동일한 학습 기회)
- (이전) Group A/B를 명확히 상충(trade-off)하도록 분리해 데이터 신호를 강화

### 4) 입력 스케일 문제 해결
- 관측에서 `x_com` 제거
- obs/act 모두 Z-score 정규화
- 시간축 균일 다운샘플(T/5 → T≈200)

### 5) 모델 구조 변경(핵심)
- RewardDecoder를 residual 구조에서 **dot-product 구조**로 전면 변경
  - \(r(t)=\langle \phi(s_t,a_t), \psi(z)\rangle\)
  - `feature_dim=16`
  - (업데이트) `psi(z)` 정규화는 **BatchNorm → LayerNorm**으로 교체(배치 통계로 z 정보가 씻기는 현상 완화)
- TrajectoryEncoder의 pooling을 max/mean에서 **attention pooling**으로 변경

### 6) 학습 안정화 설정
- `reward_scaling=200`: sigmoid 입력 \(R_1-R_2\) 스케일을 조절해 학습 신호(gradient) 품질을 확보
- `latent_dim=4`, `context_size=15` (기본값을 30→15로 조정)
- KL annealing: cosine, cycles=4
- KL weight cap: `kl_max=0.5` (annealer 출력 상한)
- Optimizer: encoder/decoder lr 분리(디코더 lr = 0.2× 인코더 lr)
- ROC 커브 저장 및 AUC 로깅

### 7) Posterior Collapse 대응 추가 업데이트 (2026-01)
아래 변경들은 “KL이 1~2 epoch 만에 0으로 붕괴”되는 현상을 타개하기 위해, **데이터(신호) → 손실/규제(인센티브) → 평가(가시화)** 순서로 적용했습니다.

#### 7.1 Disagreement pair 오버샘플링(데이터 측면 강화)
#### 7.1 오라클/데이터 생성 강화(2026-01-19): 연속 latent 유저 + 확률론 라벨 + margin 샘플링
후방 붕괴의 근본 원인이 “**전역 평균(유저 무시)으로도 꽤 맞추는 쉬운 pair가 많음**”에 있다고 판단하여,
데이터 생성 자체를 `z`가 **반드시 필요**하도록 바꾸었습니다.

- 목표
  - 쉬운 pair(Δscore가 너무 큰 비교)를 줄여 “컨텍스트 없이 맞추기 어려운” 샘플 비중을 증가
  - 2그룹 평균으로 해결되는 지름길을 차단(유저 선호를 연속 공간으로)
  - 라벨을 결정론이 아닌 Bradley–Terry 확률 모델로 생성해 난이도/노이즈를 조절 가능하게
- 변경: `scripts/build_preference_dataset.py`
  - **연속 유저 선호**: `z_user ~ N(0,I)` → `w_user`(feature-space)로 매핑 후 정규화/센터링
    - `--num_users`, `--user_latent_dim`, `--seed`
  - **확률론 라벨**: \(p(y=1)=\sigma((s_1-s_2)/T)\), `y~Bernoulli(p)`
    - `--pair_temperature`
  - **margin 기반 pair 샘플링**: `|s1-s2| ∈ [margin_min, margin_max]`를 우선적으로 만족하는 pair 생성
    - `--margin_sampling_ratio`, `--margin_min`, `--margin_max`, `--margin_max_tries`

#### 7.2 KL weight 상한(kl_max) 도입
- 목표: KL 스케줄이 과도하게 커져 학습이 불안정해지거나, 반대로 인코더가 너무 빨리 prior로 수렴하는 상황을 완화
- 변경: `scripts/train_model.py`, `src/models/vae.py`
  - `--kl_max` (기본 0.5)
  - `kl_weight = min(annealer.slope(), kl_max)` 적용

#### 7.3 Free-bits를 “진짜 KL floor(per-dim)”로 수정 + raw KL 가시화
- 관찰: 단순 `kld_loss`만 보면 free-bits/클램프에 의해 0처럼 보일 수 있어 **진짜 KL 붕괴(`kld_loss_raw`)와 분리**가 필요
- 변경 1: `src/models/vae.py`
  - free-bits를 scalar `max(KL-free_bits,0)`가 아닌, **차원별 KL floor**로 변경
  - `kld_loss_raw`: 배치 평균 \(\mathbb{E}[\sum_j KL_j]\) 유지 (로깅용)
  - `kld_loss`: per-dim 평균 KL에 대해 \(\max(KL_j, free\_bits)\) 후 합산 (collapse 방지용)
- 변경 2: `scripts/train_model.py`, `src/utils/visualization.py`
  - `training_curves.png`에 `train/eval kld_loss_raw`를 점선으로 추가 표시

#### 7.4 적응(4단계) 사용자 응답을 랜덤 → 그룹 오라클(A/B)로 전환 + 라벨 0/1 혼합
- 기존 문제: 적응 단계가 랜덤 피드백이거나, `(s1,s2)` 순서만 바꾸고 `y=1`만 넣으면 `PairEncoder`가 `y` 신호를 활용하기 어려움
- 변경: `scripts/run_interactive_adaptation.py`
  - `--oracle_group A|B` 추가: 학습 때의 두 성향(예: jerk-hater vs pitch-hater)을 가정한 오라클로 선호 응답 생성
  - pair 순서를 **(input_traj, implicit_traj)로 고정**하고, `y∈{0,1}`을 실제 값으로 저장

#### 7.5 P0: 학습↔적응/평가 전처리 완전 일치(중요)
- 진단: 학습 데이터 로더(`src/data/loader.py`)는 **x_com 제거(9D), 시간축 downsample, Z-score 정규화**를 강제하는데,
  적응/평가가 raw state/action을 그대로 사용하면 분포/차원이 달라져 목표(“few-shot 적응”)가 깨질 수 있음
- 변경
  - `src/utils/preprocessing.py` 추가: 공용 전처리/통계 저장/로드 모듈
  - `scripts/train_model.py`: 체크포인트 폴더에 `preprocessing_stats.npz` 자동 저장
  - `scripts/run_interactive_adaptation.py`: `preprocessing_stats.npz`를 로드해 동일 전처리 적용(미존재 시 에러)
  - `scripts/evaluate_adaptation.py`: 동일 전처리 적용(미존재 시 경고 후 raw 평가)

#### 7.6 P3: 적응 성능을 쌍비교 정량지표로 평가(AUC/BCE/Acc)
- 목표: “적응된 z가 실제 선호를 더 잘 맞추는지”를 trajectory 평균 보상 ranking이 아니라 **pairwise 분류 성능**으로 확인
- 변경: `scripts/evaluate_adaptation.py`
  - `--oracle_group A|B`를 주면, 오라클 기반 holdout 쌍을 샘플링해 BCE/Acc/AUC 출력
  - 옵션: `--eval_num_pairs`(기본 5000), `--eval_seed`

#### 7.7 디코더 규제: 상태/행동 feature 추출에 약한 dropout 추가
- 목표: 디코더가 너무 쉽게 지름길을 찾는 경우를 완화하여 z 경로의 필요성을 높임
- 변경: `src/models/reward_decoder.py`, `src/models/vae.py`, `scripts/train_model.py`
  - `RewardDecoder.feature_net`에 dropout(`decoder_feature_dropout`, 기본 0.1) 추가

#### 7.8 실험 기본 세팅 업데이트(현행 기본값)
- `context_size=15`
- `kl_max=0.5`
- `free_bits=0.15` (차원당 KL floor, nats/dim)
- `decoder_feature_dropout=0.1`

### 8) 다음 실험 체크리스트(권장)
- **z ablation**: z=0 고정 vs 정상 z에서 AUC/BCE 차이 확인(“z가 실제로 쓰이는지”)
- encoder/decoder gradient norm 로깅(encoder가 따라가는지 확인)
- Group A/B별 성능 분리 평가(유저별 분별력 확인)


