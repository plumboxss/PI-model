## 프로젝트 설명 (Project Description)

### 1) 문제 정의
입력 데이터는 선호 비교 데이터셋 \(D\)이며 각 샘플은 다음을 포함합니다.
- 두 궤적 \(\tau_A, \tau_B\)
- 선호 라벨 \(y \in \{0,1\}\)
- 유저/오라클 ID (`model_id`)

목표는 다음 두 모듈을 학습하는 것입니다.
- **인코더** \(q_\psi(z \mid \text{context})\): K개의 비교(context)로부터 잠재 선호 \(z\)를 추정
- **보상 디코더** \(r_\phi(s,a,z)\): 잠재 선호 \(z\)를 조건으로 보상 함수를 추정

학습은 Bradley–Terry 모델로 이진 선호를 모델링합니다.
- \(p(\tau_A \succ \tau_B) = \sigma(R_A - R_B)\)
- \(R = \sum_t r(s_t,a_t,z)\) (단, 학습 안정화를 위해 `reward_scaling`으로 스케일 조정)

### 2) 현재 데이터 생성 원리(오라클)
현재 선호 라벨은 “feature-weighted oracle”로 생성됩니다.
- 각 궤적에서 스칼라 피처 \(f(\tau)\)를 추출
  - `jerk`, `pitch`, `settling_time`, `rms_acceleration`
- 가상 유저 i의 가중치 \(w_i\)를 생성
  - Group A/B 상충 구조로 분리(음수/양수 분포 반전 포함)
- 점수: \(Score_i(\tau) = w_i \cdot \tilde f(\tau)\) (여기서 \(\tilde f\)는 StandardScaler로 정규화된 피처)
- 라벨: \(y = 1[Score_i(\tau_A) > Score_i(\tau_B)]\)

핵심은 “유저에 따라 선호 방향이 달라지도록” 구조를 강제해 z를 의미 있게 만드는 것입니다.

### 3) 모델 구조(현행)
#### 3.1 Encoder: context → z
- `TrajectoryEncoder`: \((B, T, D_{sa}) \rightarrow (B, d)\)
  - time aggregation은 **attention pooling**
- `PairEncoder`: (e1, e2, y) → h
- `SetEncoder`: \(\{h_i\}_{i=1..K} \rightarrow (\mu, \log \sigma^2)\)
- reparameterization: \(z = \mu + \sigma \odot \epsilon\)

#### 3.2 Decoder: dot-product reward
Posterior collapse를 줄이기 위해, 디코더를 오라클 구조에 맞춘 dot-product 형태로 설계합니다.
- \(\phi(s,a) \in \mathbb{R}^{16}\): 물리/행동 특징을 추출(`feature_net`)
- \(\psi(z) \in \mathbb{R}^{16}\): 유저 선호 가중치를 생성(`weight_net`)
- \(r(t) = \langle \phi(s_t,a_t), \psi(z)\rangle\)

`weight_net` 출력에는 BatchNorm을 적용해 스케일을 안정화하고, z 경로의 학습 신호가 약해지지 않도록 돕습니다.

### 4) 입력 스케일링/시퀀스 길이
#### 4.1 x_com 제거
관측(state)에서 절대 위치(`x_com`)는 선호와 무관하고 스케일이 커서 모델을 압도할 수 있어 제거합니다.

#### 4.2 obs/act 정규화
obs와 act에 대해 Z-score 정규화를 적용합니다.

#### 4.3 시간축 다운샘플
T=1000은 학습 안정성과 계산량 측면에서 크기 때문에, 균일 다운샘플(1/5)로 T≈200을 사용합니다.

### 5) 평가/시각화
- 학습 곡선(`training_curves.png`)
- ROC 커브(`roc_epoch_{epoch}.png`) 및 `eval/roc_auc` 로깅


