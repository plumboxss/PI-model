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
- feature-weighted oracle로 라벨링 전환
- 유저별 쿼터를 균등 분배(각 유저가 동일한 학습 기회)
- Group A/B를 명확히 상충(trade-off)하도록 분리하고, 분포를 음수/양수로 반전하여 차이를 강화

### 4) 입력 스케일 문제 해결
- 관측에서 `x_com` 제거
- obs/act 모두 Z-score 정규화
- 시간축 균일 다운샘플(T/5 → T≈200)

### 5) 모델 구조 변경(핵심)
- RewardDecoder를 residual 구조에서 **dot-product 구조**로 전면 변경
  - \(r(t)=\langle \phi(s_t,a_t), \psi(z)\rangle\)
  - `feature_dim=16`
  - `weight_net` 출력에 BatchNorm 적용(tanh 제거)
- TrajectoryEncoder의 pooling을 max/mean에서 **attention pooling**으로 변경

### 6) 학습 안정화 설정
- `reward_scaling=1000`: sigmoid saturation 완화
- `latent_dim=4`, `context_size=30`
- KL annealing: cosine, cycles=4
- Optimizer: encoder/decoder lr 분리(디코더 lr = 0.2× 인코더 lr)
- ROC 커브 저장 및 AUC 로깅

### 7) 다음 실험 체크리스트(권장)
- **z ablation**: z=0 고정 vs 정상 z에서 AUC/BCE 차이 확인(“z가 실제로 쓰이는지”)
- encoder/decoder gradient norm 로깅(encoder가 따라가는지 확인)
- Group A/B별 성능 분리 평가(유저별 분별력 확인)


