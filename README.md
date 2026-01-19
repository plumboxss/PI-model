## 차량 서스펜션 제어를 위한 선호도 기반 보상 모델

### 1. 개요

이 프로젝트는 시뮬레이션된 차량의 서스펜션을 제어하기 위한 선호도 기반 보상 학습 시스템을 구현하고 평가합니다. 핵심 아이디어는 소수의 궤적 샘플 간 쌍대 비교 데이터로부터 잠재 선호도 벡터 `z`를 학습하고, 이를 조건부 보상 모델에 주입해 사용자 맞춤 보상을 추론하는 것입니다. 최근 업데이트로 다음이 반영되었습니다:

- 시뮬레이션 다양성: PD+shaping 제어기(`kp/kd/shaping_factor`)와 범프 높이/폭 랜덤화, 승차감 지표(`rms_acceleration`) 추가
- 데이터셋(업데이트): **연속(latent) 유저 선호 + 확률론(Bradley–Terry) 라벨 + margin 기반 pair 샘플링**으로 “쉬운 pair”를 줄여 `z`가 의미 있게 쓰이도록 데이터 신호를 강화
- 모델: VAE 기반 잠재 선호 벡터 `z` 추론 + 선호도 조건부 보상 모델 `r(s, a, z)`
- 학습(코드 기본값): `context_size=15`, `annealer_cycles=4`, `kl_max=0.5`

자세한 방법론/설계 의도는 `PROJECT_DESCRIPTION.md`와 `RESEARCH_LOG.md`를 참고하세요.

### 2. 핵심 개념

- **잠재 사용자 임베딩 모델 (`q_ψ`)**: 궤적 비교 데이터셋 `D = { (σ_A, σ_B, y) }`으로부터 잠재 선호도 벡터 `z`를 추론하는 인코더입니다. 이 모델은 분포 `q(z|D)`를 학습합니다.
- **선호도 조건부 보상 모델 (`r_φ`)**: 보상 함수 역할을 하는 디코더입니다. 현재 상태(state), 행동(action), 그리고 추론된 잠재 선호도 벡터 `z`를 기반으로 스칼라 보상 `r`을 예측합니다. 즉, `r = r_φ(s, a, z)` 입니다.
- **VAE 프레임워크**: 인코더와 디코더는 Variational Autoencoder(VAE) 구조 안에서 함께 학습되어, 주어진 선호도 데이터셋에 대한 예측 정확도를 높입니다.
- **빠른 적응 (Fast Adaptation)**: 사전 학습이 완료된 후, 모델은 소수의 새로운 비교 쿼리만으로 새로운 사용자 선호도에 해당하는 `z`를 신속하게 추론하여 새로운 환경에 빠르게 적응할 수 있습니다.

### 3. 설치 및 환경 설정

모든 의존성은 Conda를 통해 관리됩니다.

```bash
conda env create -f environment.yml
conda activate pi-model
```

### 4. 전체 워크플로우

프로젝트는 원본 데이터 생성부터 적응된 보상 함수 평가까지 총 5단계의 워크플로우를 따릅니다.

#### 1단계: 원본 궤적 데이터 생성

먼저, 차량 시뮬레이션을 실행하여 기본적인 궤적 데이터셋을 생성합니다. PD 게인(`kp/kd`), shaping, 범프 높이/폭을 랜덤화해 “빠르지만 거친” vs “느리지만 부드러운” 궤적을 함께 생성합니다. 궤적 피처에는 `jerk`, `pitch`, `settling_time`(최종값 밴드+유지시간 기준), `rms_acceleration`이 포함됩니다.

```bash
# 예시: 500개의 궤적 생성 (시각화 포함)
python scripts/generate_data.py \
  --num-episodes 500 \
  --dataset-id A \
  --dataset-name raw_trajectories_A \
  --visualize
```

이 명령은 `artifacts/A/datasets/raw_trajectories_A.pkl` 파일을 생성하고, 시각화는 `artifacts/A/visualizations/`에 저장됩니다.

#### 2단계: 선호도 데이터셋 구축

다음으로, 생성된 원본 궤적들을 선호도 데이터셋으로 가공합니다.

- **유저 선호(연속 latent)**: 유저별 잠재 벡터 `z_user ~ N(0, I)`를 샘플링하고, 이를 feature-space 가중치 `w_user`로 매핑하여(정규화/센터링) 유저 선호가 연속적으로 변하도록 만듭니다.
- **확률론 라벨(Bradley–Terry)**: 결정론 `1[score1>=score2]` 대신,
  - \(p(y=1)=\sigma((score_1-score_2)/T)\) 로 두고 `Bernoulli(p)`로 라벨을 생성합니다(`T=pair_temperature`).
- **margin 기반 pair 샘플링**: `|score1-score2|`가 특정 범위에 들어오는 pair를 우선적으로 생성해 **너무 쉬운 pair(전역 평균으로 맞추는 지름길)** 비중을 줄입니다.

```bash
python scripts/build_preference_dataset.py \
  --input_path artifacts/A/datasets/raw_trajectories_A.pkl \
  --output_path datasets/preference_dataset_A.pkl \
  --num_pairs 20000 \
  --seed 42 \
  --num_users 200 \
  --user_latent_dim 8 \
  --pair_temperature 0.5 \
  --margin_sampling_ratio 0.9 \
  --margin_min 0.2 \
  --margin_max 1.5 \
  --margin_max_tries 80 \
  --visualize
```
python scripts/build_preference_dataset.py \
  --input_path artifacts/A/datasets/raw_trajectories_A.pkl \
  --output_path datasets/preference_dataset_A_prob_margin_u50_p25k.pkl \
  --num_pairs 25000 \
  --seed 42 \
  --num_users 50 \
  --user_latent_dim 8 \
  --pair_temperature 0.5 \
  --margin_sampling_ratio 0.9 \
  --margin_min 0.2 \
  --margin_max 1.5 \
  --margin_max_tries 80 \
  --visualize
시각화는 `datasets/visualizations/`에 저장됩니다.

**(참고) 핵심 플래그 요약**

- `--pair_temperature`: 작을수록 결정론에 가까워지고, 클수록 라벨 노이즈가 커집니다(너무 작으면 쉬운 데이터, 너무 크면 학습 불가능).
- `--margin_*`: 쉬운 pair를 줄이는 핵심. `margin_min`을 올리면 더 어려워지고, `margin_max`를 너무 크게 하면 쉬운 pair가 다시 섞입니다.

#### 3단계: VAE 모델 사전 학습

이전 단계에서 생성한 선호도 데이터셋으로 VAE 모델(인코더 `q_ψ`, 보상 디코더 `r_φ`)을 학습시킵니다. 학습 과정은 Weights & Biases를 통해 기록되며, 학습 곡선은 로컬 파일로도 저장됩니다.

- 코드 기본값(현재 `scripts/train_model.py` 기준): `context_size=15`, `annealer_cycles=4`, `trajectory_encoder_type=mlp`, `set_encoder_type=attention`, `reward_scaling=200`, `free_bits=0.15`, `kl_max=0.5`, `decoder_feature_dropout=0.1`
- 실험에서 다른 값을 쓰려면 아래처럼 플래그로 명시하세요(예: `context_size=15`, `annealer_cycles=1`)

```bash
# 먼저 wandb에 로그인해야 합니다: wandb login
python scripts/train_model.py \
  --dataset_path datasets/preference_dataset_A.pkl \
  --logging.output_dir "logs" \
  --comment "pretrain_suspension_model_A" \
  --seed 42 \
  --latent_dim 8 \
  --context_size 15 \
  --annealer_cycles 4 \
  --kl_max 0.5 \
  --free_bits 0.15 \
  --save_training_curves True
```
python scripts/train_model.py \
  --dataset_path datasets/preference_dataset_A_prob_margin_viz.pkl \
  --logging.output_dir "logs" \
  --comment "pretrain_suspension_model_A" \
  --seed 42 \
  --latent_dim 4 \
  --context_size 15 \
  --annealer_cycles 4 \
  --kl_max 0.5 \
  --free_bits 0.15 \
  --save_training_curves True

학습된 모델(`model.pt` 등)이 `logs/` 디렉토리 내부에 저장되고, 학습 곡선은 `logs/{env}/{model_type}/{comment}/s{seed}/training_curves.png`에 자동 저장됩니다.

**(중요) 전처리 일치(P0): 학습 통계 저장**

학습 시 `ContextQueryDataset` 기준으로 다음 전처리가 적용됩니다: **x_com 제거(9D), 시간축 다운샘플, obs/act Z-score 정규화**.  
적응/평가 단계도 동일 전처리를 적용하기 위해, 학습 스크립트는 체크포인트 폴더에 아래 파일을 자동 저장합니다:

- `preprocessing_stats.npz`: `obs_mean/std`, `act_mean/std`, `downsample_step`, `obs_dim_used` 등

**(추가) KL weight cap (`--kl_max`)**

KL annealing이 너무 강해지면 KL term이 과도하게 커져 학습이 불안정해지거나, 반대로 인코더가 “안 쓰이는 방향”으로 수렴하는 문제가 생길 수 있습니다. 이를 완화하기 위해 **annealer가 산출한 KL weight에 상한을 두는 옵션**을 제공합니다.

- `--kl_max`: KL weight 상한 (기본 `0.5`)

#### 4단계: 상호작용을 통한 적응

사전 학습된 모델을 특정 선호도에 맞게 적응시키는 상호작용 세션을 시뮬레이션합니다. 이 스크립트는 모델을 로드하고, 사용자에게 질문할 궤적 쌍을 선택한 뒤, 응답을 받아 잠재 벡터 `z`를 업데이트합니다.

```bash
# 예시: 적응 루프 실행 (시각화 포함)
python scripts/run_interactive_adaptation.py \
  --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
  --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
  --preprocess_stats_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/preprocessing_stats.npz" \
  --output_z_path "data/adapted_z.pt" \
  --comparison_set_size 1000 \
  --diversity_epsilon 0.1 \
  --oracle_group A \
  --visualize
```

이 스크립트는 최종적으로 적응된 잠재 벡터를 담은 `adapted_z.pt` 파일을 출력하고, 시각화는 `data/visualizations/`에 저장됩니다.

**(구현 디테일) 적응 단계의 사용자 피드백(오라클 시뮬레이션)**

- 적응 단계의 “사용자”는 랜덤이 아니라, 학습 때의 두 사용자 성향을 가정한 **그룹 오라클**로 시뮬레이션됩니다.
  - `--oracle_group A`: jerk/rms_acceleration를 강하게 싫어하는 사용자
  - `--oracle_group B`: pitch/settling_time을 강하게 선호하는 사용자
- 매 스텝에서 입력 궤적(1개)과 모델이 고른 implicit 궤적(1개)으로 **쌍 비교**를 만들고,
  - pair 순서는 **(input_traj, implicit_traj)로 고정**
  - 컨텍스트 라벨 `y`는 오라클의 선호에 따라 **0/1이 실제로 섞이게 저장**됩니다.
    - `y=1`이면 `input_traj ≻ implicit_traj`
    - `y=0`이면 `implicit_traj ≻ input_traj`

이렇게 쌍의 순서를 고정하고 `y`를 실제 값으로 주면, 적응 인코더(`PairEncoder`)가 `y` 신호를 제대로 활용할 수 있어(상수 `y=1` 문제 제거) `z` 재추론이 더 안정적입니다.

#### 5단계: 적응된 보상 함수 평가

마지막으로, 적응된 잠재 벡터 `z`를 사용하여 개인화된 새로운 보상 함수 `r_new(s, a) = r_φ(s, a, z_adapted)`를 정의합니다.

```bash
# 기본 평가
python scripts/evaluate_adaptation.py \
  --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
  --adapted_z_path "data/adapted_z.pt" \
  --preprocess_stats_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/preprocessing_stats.npz"

# 궤적 점수 계산 및 시각화 포함
python scripts/evaluate_adaptation.py \
  --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
  --adapted_z_path "data/adapted_z.pt" \
  --preprocess_stats_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/preprocessing_stats.npz" \
  --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
  --visualize

# 적응 전/후 비교 (before_z_path 제공 시)
python scripts/evaluate_adaptation.py \
  --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
  --adapted_z_path "data/adapted_z.pt" \
  --preprocess_stats_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/preprocessing_stats.npz" \
  --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
  --before_z_path "data/initial_z.pt" \
  --visualize

**(추가, P3) 쌍비교 성능 평가(AUC/BCE/Acc)**

적응된 `z`가 실제로 선호를 잘 맞추는지 확인하려면, `oracle_group`를 지정해 **오라클 기반 holdout 쌍비교**를 생성하고 AUC/BCE/Acc를 출력할 수 있습니다:

```bash
python scripts/evaluate_adaptation.py \
  --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
  --adapted_z_path "data/adapted_z.pt" \
  --preprocess_stats_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/preprocessing_stats.npz" \
  --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
  --oracle_group A \
  --eval_num_pairs 5000
```
```

시각화는 `data/visualizations/`에 저장됩니다. 이 스크립트는 새롭게 정의된 보상 함수를 후속 강화학습이나 추가 분석에 사용할 수 있도록 하는 기반을 제공합니다.

### 5. 시각화

프로젝트 전체 워크플로우에서 성과를 확인할 수 있는 종합 시각화 시스템이 포함되어 있습니다.

#### 시각화 생성 위치

- **데이터 생성**: `artifacts/{dataset_id}/visualizations/`
  - `trajectory_samples.png`: 궤적 샘플 (6개)
  - `feature_distributions.png`: 특징 분포 (`jerk`, `pitch`, `settling_time`, `rms_acceleration`)
- **선호도 데이터셋**: `datasets/visualizations/`
  - `preference_distribution.png`: 라벨/유저별 pair 수 분포 요약
  - `preference_oracle_diagnostics.png`: 오라클 진단(난이도 |Δscore|, p(y=1) 분포, P(y=1|Δscore) 정합성, 유저별 라벨 편향)
  - `user_weight_heatmap.png`: 유저 선호 가중치(oracle `w_user`) 히트맵
- **모델 학습**: `logs/{env}/{model_type}/{comment}/s{seed}/`
  - `training_curves.png`: 학습 곡선 (Loss, Accuracy, KL Divergence(`kld_loss` + `kld_loss_raw`), KL Weight)
- **적응 단계**: `data/visualizations/`
  - `z_evolution.png`: z 벡터 진화 (각 차원별 변화, L2 norm)
  - `adaptation_summary.png`: 적응 요약 (PCA, context 크기, 변화율)
- **평가 단계**: `data/visualizations/`
  - `reward_distribution.png`: 보상 분포 (전체 분포, 상위 궤적)
  - `before_after_comparison.png`: 적응 전/후 비교 (분포 비교, 산점도, 변화량, 통계 요약)

#### 시각화 사용법 요약

각 단계에서 `--visualize` 플래그를 추가하면 자동으로 시각화가 생성됩니다.

```bash
# 1단계: 데이터 생성 시각화
python scripts/generate_data.py --num-episodes 500 --dataset-id A --visualize

# 2단계: 선호도 데이터셋 시각화
python scripts/build_preference_dataset.py \
  --input_path artifacts/A/datasets/raw_trajectories_A.pkl \
  --output_path datasets/preference_dataset_A.pkl \
  --num_pairs 20000 \
  --seed 42 \
  --num_users 200 \
  --user_latent_dim 8 \
  --pair_temperature 0.5 \
  --margin_sampling_ratio 0.9 \
  --margin_min 0.2 \
  --margin_max 1.5 \
  --margin_max_tries 80 \
  --visualize

# 3단계: 학습 곡선 자동 저장 (--save_training_curves True, 기본값)
python scripts/train_model.py \
  --dataset_path datasets/preference_dataset_A.pkl \
  --save_training_curves True \
  --comment "pretrain_suspension_model_A"

# 4단계: 적응 과정 시각화
python scripts/run_interactive_adaptation.py \
  --vae_model_path "logs/.../best_model.pt" \
  --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
  --output_z_path "data/adapted_z.pt" \
  --visualize

# 5단계: 평가 시각화
python scripts/evaluate_adaptation.py \
  --vae_model_path "logs/.../best_model.pt" \
  --adapted_z_path "data/adapted_z.pt" \
  --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
  --visualize
```

자세한 내용은 `VISUALIZATION_GUIDE.md`를 참고하세요.

### 6. 파일 구조

- **`scripts/`**: 메인 워크플로우를 위한 실행 스크립트
  - `generate_data.py`: (1단계) 차량 시뮬레이션 실행
  - `build_preference_dataset.py`: (2단계) 원본 궤적을 선호도 데이터셋으로 가공
  - `train_model.py`: (3단계) VAE 모델 학습
  - `run_interactive_adaptation.py`: (4단계) 상호작용 쿼리 및 적응 시뮬레이션
  - `evaluate_adaptation.py`: (5단계) 적응된 `z`를 사용해 새로운 보상 함수 평가
- **`src/`**: 기능별 핵심 소스 코드
  - `simulation/`: 시뮬레이션 핵심 요소 (`env.py`, `plant.py`, `vehicle_model.py` 등)
  - `models/`: VAE 모델 아키텍처 (`vae.py`, `trajectory_encoder.py`, `pair_encoder.py`, `set_encoder.py`, `reward_decoder.py`)
  - `data/`: 데이터 로딩 및 처리 로직 (`loader.py`)
  - `utils/`: 학습/시각화/시뮬레이션 유틸 (`visualization.py`, `training_utils.py` 등)
- **`configs/`**: 설정 파일 (`.yaml`)
- **`environment.yml`**: Conda 가상 환경 의존성
- **`requirements.txt`**: pip 의존성
- **`PROJECT_DESCRIPTION.md`**: 프로젝트 목표/오라클/모델/학습 정의
- **`RESEARCH_LOG.md`**: 연구 과정 및 실험 노트
