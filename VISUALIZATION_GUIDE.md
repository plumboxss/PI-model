# 프로젝트 전체 시각화 가이드

프로젝트 전체 워크플로우에서 성과를 확인할 수 있는 종합 시각화 시스템입니다.

## 📊 시각화 구성 요소

### 1. 데이터 생성 단계 (`generate_data.py`)

**기능:**
- 궤적 샘플 시각화 (6개 랜덤 샘플)
- 특징 분포 시각화 (jerk, pitch, settling_time)

**사용법:**
```bash
python scripts/generate_data.py \
    --num-episodes 1000 \
    --dataset-id A \
    --visualize
```

**생성 파일:**
- `artifacts/A/visualizations/trajectory_samples.png`
- `artifacts/A/visualizations/feature_distributions.png`

---

### 2. 선호도 데이터셋 구축 단계 (`build_preference_dataset.py`)

**기능:**
- 클러스터링 결과 시각화 (PCA 투영, 클러스터 크기)
- 선호도 쌍 분포 시각화 (label 분포, user group별 분포)

**사용법:**
```bash
python scripts/build_preference_dataset.py \
    --input_path artifacts/A/datasets/1000.pkl \
    --output_path datasets/preference_dataset_A.pkl \
    --num_clusters 16 \
    --num_pairs 20000 \
    --visualize
```

**생성 파일:**
- `datasets/visualizations/clustering_results.png`
- `datasets/visualizations/preference_distribution.png`

---

### 3. 모델 학습 단계 (`train_model.py`)

**기능:**
- 학습 곡선 시각화 (Loss, Accuracy, KL Divergence, KL Weight)
- 로컬 파일로 저장 (WandB 의존성 없음)

**사용법:**
```bash
python scripts/train_model.py \
    --dataset_path datasets/preference_dataset_A.pkl \
    --save_training_curves True \
    ...
```

**생성 파일:**
- `logs/{env}/{model_type}/{comment}/s{seed}/training_curves.png`

**자동 저장:**
- 마지막 epoch 또는 `save_freq` 주기마다 자동 저장

---

### 4. 적응 단계 (`run_interactive_adaptation.py`)

**기능:**
- z 벡터 진화 시각화 (각 차원별 변화, L2 norm)
- 적응 요약 시각화 (PCA, context 크기, 변화율)

**사용법:**
```bash
python scripts/run_interactive_adaptation.py \
    --vae_model_path "logs/.../best_model.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/1000.pkl" \
    --output_z_path "data/adapted_z.pt" \
    --visualize
```

**생성 파일:**
- `data/visualizations/z_evolution.png`
- `data/visualizations/adaptation_summary.png`

---

### 5. 평가 단계 (`evaluate_adaptation.py`)

**기능:**
- 보상 분포 시각화 (전체 분포, 상위 궤적)
- 적응 전/후 비교 시각화 (분포 비교, 산점도, 변화량, 통계 요약)

**사용법:**
```bash
python scripts/evaluate_adaptation.py \
    --vae_model_path "logs/.../best_model.pt" \
    --adapted_z_path "data/adapted_z.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/1000.pkl" \
    --visualize \
    --before_z_path "data/initial_z.pt"  # optional: 적응 전 z 벡터
```

**생성 파일:**
- `data/visualizations/reward_distribution.png`
- `data/visualizations/before_after_comparison.png` (before_z_path 제공 시)

---

## 🎯 전체 워크플로우 시각화

### 단계별 시각화 요약

```
1. 데이터 생성
   └─> artifacts/{dataset_id}/visualizations/
       ├─ trajectory_samples.png
       └─ feature_distributions.png

2. 선호도 데이터셋 구축
   └─> datasets/visualizations/
       ├─ clustering_results.png
       └─ preference_distribution.png

3. 모델 학습
   └─> logs/{env}/{model_type}/{comment}/s{seed}/
       └─ training_curves.png

4. 적응 단계
   └─> data/visualizations/
       ├─ z_evolution.png
       └─ adaptation_summary.png

5. 평가 단계
   └─> data/visualizations/
       ├─ reward_distribution.png
       └─ before_after_comparison.png
```

---

## 📈 시각화 해석 가이드

### 데이터 생성 단계
- **trajectory_samples**: 생성된 궤적의 다양성 확인
- **feature_distributions**: 데이터 품질 확인 (jerk, pitch, settling_time 분포)

### 선호도 데이터셋 구축
- **clustering_results**: 사용자 그룹 분리 정도 확인
- **preference_distribution**: 선호도 쌍의 균형 확인

### 모델 학습
- **training_curves**: 
  - Loss가 감소하는지 확인
  - Accuracy가 향상되는지 확인
  - KL Divergence가 적절히 유지되는지 확인
  - KL Weight annealing이 올바르게 작동하는지 확인

### 적응 단계
- **z_evolution**: z 벡터가 수렴하는지 확인
- **adaptation_summary**: 
  - Context 증가에 따른 z 변화 확인
  - 적응이 수렴하는지 확인 (변화율 감소)

### 평가 단계
- **reward_distribution**: 적응된 보상 함수의 궤적 평가 분포
- **before_after_comparison**: 
  - 적응 전/후 보상 분포 비교
  - 개선된 궤적 비율 확인
  - 평균 보상 변화 확인

---

## 🔧 프로그래밍 방식 사용

시각화 함수를 직접 호출하여 커스텀 시각화를 만들 수 있습니다:

```python
from src.utils.visualization import (
    plot_trajectory_samples,
    plot_feature_distributions,
    plot_clustering_results,
    plot_preference_distribution,
    plot_training_curves,
    plot_z_evolution,
    plot_adaptation_summary,
    plot_reward_distribution,
    plot_before_after_comparison
)

# 예시: 학습 곡선 시각화
metrics_history = {
    'train/loss': [...],
    'eval/loss': [...],
    'train/accuracy': [...],
    ...
}
plot_training_curves(metrics_history, save_path='my_curves.png')
```

---

## 📁 파일 구조

```
src/utils/
  └─ visualization.py  # 모든 시각화 함수

artifacts/{dataset_id}/
  └─ visualizations/
      ├─ trajectory_samples.png
      └─ feature_distributions.png

datasets/
  └─ visualizations/
      ├─ clustering_results.png
      └─ preference_distribution.png

logs/{env}/{model_type}/{comment}/s{seed}/
  └─ training_curves.png

data/visualizations/
  ├─ z_evolution.png
  ├─ adaptation_summary.png
  ├─ reward_distribution.png
  └─ before_after_comparison.png
```

---

## ✅ 체크리스트

프로젝트 전체 성과 확인을 위한 체크리스트:

- [ ] 데이터 생성: `generate_data.py --visualize`
- [ ] 선호도 데이터셋: `build_preference_dataset.py --visualize`
- [ ] 모델 학습: `train_model.py --save_training_curves True`
- [ ] 적응 단계: `run_interactive_adaptation.py --visualize`
- [ ] 평가 단계: `evaluate_adaptation.py --visualize`

모든 단계의 시각화를 생성하면 프로젝트 전체 성과를 완전히 확인할 수 있습니다!

