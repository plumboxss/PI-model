# 차량 서스펜션 제어를 위한 선호도 기반 보상 모델

## 1. 개요

이 프로젝트는 시뮬레이션된 차량의 서스펜션을 제어하기 위한 선호도 기반 보상 학습 시스템을 구현하고 평가합니다. 핵심 아이디어는 소수의 궤적 샘플 간 쌍대 비교 데이터로부터 사용자의 잠재적인 선호도 표현을 학습하는 것입니다. 이렇게 학습된 잠재 선호도 벡터는 보상 모델을 조건화(condition)하여, 특정 사용자의 요구(예: 부드러운 승차감 선호 vs. 스포티한 핸들링 선호)에 맞게 차량의 제어 동작을 신속하게 적응시킬 수 있도록 합니다.

모든 방법론은 `project_intent_description.txt` 문서에 기술된 개념을 기반으로 합니다.

## 2. 핵심 개념

*   **잠재 사용자 임베딩 모델 (`q_ψ`)**: 궤적 비교 데이터셋 `D = { (σ_A, σ_B, y) }`으로부터 잠재 선호도 벡터 `z`를 추론하는 Transformer 기반 인코더입니다. 이 모델은 분포 `q(z|D)`를 학습합니다.
*   **선호도 조건부 보상 모델 (`r_φ`)**: 보상 함수 역할을 하는 디코더입니다. 현재 상태(state), 행동(action), 그리고 추론된 잠재 선호도 벡터 `z`를 기반으로 스칼라 보상 `r`을 예측합니다. 즉, `r = r_φ(s, a, z)` 입니다.
*   **VAE 프레임워크**: 인코더와 디코더는 Variational Autoencoder(VAE) 구조 안에서 함께 학습되어, 주어진 선호도 데이터셋에 대한 예측 정확도를 높입니다.
*   **빠른 적응 (Fast Adaptation)**: 사전 학습이 완료된 후, 모델은 소수의 새로운 비교 쿼리만으로 새로운 사용자 선호도에 해당하는 `z`를 신속하게 추론하여 새로운 환경에 빠르게 적응할 수 있습니다.

## 3. 설치 및 환경 설정

모든 의존성은 Conda를 통해 관리됩니다.

1.  **Conda 가상 환경 생성 및 활성화:**
    ```bash
    conda env create -f environment.yml
    conda activate pi-model
    ```

## 4. 전체 워크플로우

프로젝트는 원본 데이터 생성부터 적응된 보상 함수 평가까지 총 5단계의 워크플로우를 따릅니다.

### 1단계: 원본 궤적 데이터 생성

먼저, 차량 시뮬레이션을 실행하여 기본적인 궤적 데이터셋을 생성합니다. 시뮬레이션은 다양한 P-제어기(Proportional controller) 게인 값을 사용하여 다채로운 주행 패턴을 만들어냅니다.

```bash
# 예시: 500개의 궤적 생성 (시각화 포함)
python scripts/generate_data.py \
    --num-episodes 500 \
    --dataset-id A \
    --dataset-name raw_trajectories_A \
    --visualize
```
python scripts/generate_data.py --num-episodes 500 --dataset-id A --dataset-name raw_trajectories_A --visualize
*이 명령은 `artifacts/A/datasets/raw_trajectories_A.pkl` 파일을 생성하고, 시각화는 `artifacts/A/visualizations/`에 저장됩니다.*

### 2단계: 선호도 데이터셋 구축

다음으로, 생성된 원본 궤적들을 선호도 데이터셋으로 가공합니다. 이 스크립트는 특징 기반 클러스터링(K-Means)을 사용하여 다양한 사용자 그룹을 시뮬레이션하고, 각 클러스터별 점수 함수에 따라 쌍대 비교 레이블을 생성합니다.

```bash
# 예시: 16개의 클러스터와 20,000개의 선호도 쌍으로 데이터셋 생성 (시각화 포함)
python scripts/build_preference_dataset.py \
    --input_path artifacts/A/datasets/raw_trajectories_A.pkl \
    --output_path datasets/preference_dataset_A.pkl \
    --num_clusters 16 \
    --num_pairs 20000 \
    --visualize
```
*시각화는 `datasets/visualizations/`에 저장됩니다.*

### 3단계: VAE 모델 사전 학습

이전 단계에서 생성한 선호도 데이터셋으로 VAE 모델(인코더 `q_ψ`와 디코더 `r_φ`)을 학습시킵니다. 학습 과정은 Weights & Biases를 통해 기록되며, 학습 곡선은 로컬 파일로도 저장됩니다.

```bash
# 먼저 wandb에 로그인해야 합니다: wandb login
python scripts/train_model.py \
    --dataset_path datasets/preference_dataset_A.pkl \
    --logging.output_dir "logs" \
    --comment "pretrain_suspension_model_A" \
    --seed 42 \
    --latent_dim 8 \
    --context_size 5 \
    --save_training_curves True
```

*학습된 모델 (`model.pt` 등)이 `logs/` 디렉토리 내부에 저장되고, 학습 곡선은 `logs/{env}/{model_type}/{comment}/s{seed}/training_curves.png`에 자동 저장됩니다.*

### 4단계: 상호작용을 통한 적응

사전 학습된 모델을 특정 선호도에 맞게 적응시키는 상호작용 세션을 시뮬레이션합니다. 이 스크립트는 모델을 로드하고, 사용자에게 질문할 궤적 쌍을 지능적으로 선택한 뒤, 응답을 받아 잠재 벡터 `z`를 업데이트합니다.

```bash
# 예시: 적응 루프 실행 (시각화 포함)
python scripts/run_interactive_adaptation.py \
    --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
    --output_z_path "data/adapted_z.pt" \
    --comparison_set_size 1000 \
    --diversity_epsilon 0.1 \
    --visualize
```
*이 스크립트는 최종적으로 적응된 잠재 벡터를 담은 `adapted_z.pt` 파일을 출력하고, 시각화는 `data/visualizations/`에 저장됩니다.*

### 5단계: 적응된 보상 함수 평가

마지막으로, 적응된 잠재 벡터 `z`를 사용하여 개인화된 새로운 보상 함수 `r_new(s, a) = r_φ(s, a, z_adapted)`를 정의합니다.

```bash
# 기본 평가
python scripts/evaluate_adaptation.py \
    --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
    --adapted_z_path "data/adapted_z.pt"

# 궤적 점수 계산 및 시각화 포함
python scripts/evaluate_adaptation.py \
    --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
    --adapted_z_path "data/adapted_z.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
    --visualize

# 적응 전/후 비교 (before_z_path 제공 시)
python scripts/evaluate_adaptation.py \
    --vae_model_path "logs/Suspension-v0/VAE/pretrain_suspension_model_A/s42/best_model.pt" \
    --adapted_z_path "data/adapted_z.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/raw_trajectories_A.pkl" \
    --before_z_path "data/initial_z.pt" \
    --visualize
```
*시각화는 `data/visualizations/`에 저장됩니다. 이 스크립트는 새롭게 정의된 보상 함수를 후속 강화학습이나 추가 분석에 사용할 수 있도록 하는 기반을 제공합니다.*

## 5. 시각화

프로젝트 전체 워크플로우에서 성과를 확인할 수 있는 종합 시각화 시스템이 포함되어 있습니다.

### 시각화 생성 위치

- **데이터 생성**: `artifacts/{dataset_id}/visualizations/`
  - `trajectory_samples.png`: 궤적 샘플 (6개)
  - `feature_distributions.png`: 특징 분포 (jerk, pitch, settling_time)

- **선호도 데이터셋**: `datasets/visualizations/`
  - `clustering_results.png`: 클러스터링 결과 (PCA 투영, 클러스터 크기)
  - `preference_distribution.png`: 선호도 쌍 분포

- **모델 학습**: `logs/{env}/{model_type}/{comment}/s{seed}/`
  - `training_curves.png`: 학습 곡선 (Loss, Accuracy, KL Divergence, KL Weight)

- **적응 단계**: `data/visualizations/`
  - `z_evolution.png`: z 벡터 진화 (각 차원별 변화, L2 norm)
  - `adaptation_summary.png`: 적응 요약 (PCA, context 크기, 변화율)

- **평가 단계**: `data/visualizations/`
  - `reward_distribution.png`: 보상 분포 (전체 분포, 상위 궤적)
  - `before_after_comparison.png`: 적응 전/후 비교 (분포 비교, 산점도, 변화량, 통계 요약)

### 시각화 사용법 요약

각 단계에서 `--visualize` 플래그를 추가하면 자동으로 시각화가 생성됩니다:

```bash
# 1단계: 데이터 생성 시각화
python scripts/generate_data.py --num-episodes 500 --dataset-id A --visualize

# 2단계: 선호도 데이터셋 시각화
python scripts/build_preference_dataset.py \
    --input_path artifacts/A/datasets/500.pkl \
    --output_path datasets/preference_dataset_A.pkl \
    --num_clusters 16 \
    --num_pairs 20000 \
    --visualize

# 3단계: 학습 곡선 자동 저장 (--save_training_curves True, 기본값)
python scripts/train_model.py \
    --dataset_path datasets/preference_dataset_A.pkl \
    --save_training_curves True \
    ...

# 4단계: 적응 과정 시각화
python scripts/run_interactive_adaptation.py \
    --vae_model_path "logs/.../best_model.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/500.pkl" \
    --output_z_path "data/adapted_z.pt" \
    --visualize

# 5단계: 평가 시각화
python scripts/evaluate_adaptation.py \
    --vae_model_path "logs/.../best_model.pt" \
    --adapted_z_path "data/adapted_z.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/500.pkl" \
    --visualize
```

자세한 내용은 `VISUALIZATION_GUIDE.md`를 참고하세요.

## 6. 파일 구조

- **`scripts/`**: 메인 워크플로우를 위한 모든 실행용 스크립트.
  - `generate_data.py`: (1단계) 차량 시뮬레이션 실행.
  - `build_preference_dataset.py`: (2단계) 원본 궤적을 선호도 데이터셋으로 가공.
  - `train_model.py`: (3단계) VAE 모델 학습.
  - `run_interactive_adaptation.py`: (4단계) 상호작용 쿼리 및 적응 시뮬레이션.
  - `evaluate_adaptation.py`: (5단계) 적응된 `z`를 사용하여 새로운 보상 함수 생성.
- **`src/`**: 기능별로 정리된 모든 핵심 소스 코드.
  - `simulation/`: 시뮬레이션 핵심 요소 (`env.py`, `plant.py`, `vehicle_model.py` 등).
  - `models/`: VAE 모델 아키텍처 (`vae.py`, `trajectory_encoder.py`, `pair_encoder.py`, `set_encoder.py`, `reward_decoder.py`).
  - `data/`: 데이터 로딩 및 처리 로직 (`loader.py`).
  - `utils/`: 학습, 시각화, 시뮬레이션을 위한 유틸리티 함수 (`visualization.py`, `training_utils.py` 등).
- **`configs/`**: 모든 설정 파일 (`.yaml`).
- **`environment.yml`**: Conda 가상 환경 의존성 파일.
- **`requirements.txt`**: Pip 의존성 파일.
- **`project_intent_description.txt`**: 프로젝트의 상세 방법론을 기술한 문서.
