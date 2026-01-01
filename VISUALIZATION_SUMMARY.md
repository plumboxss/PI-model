# 프로젝트 전체 시각화 구성 요소 요약

## ✅ 완료된 작업

프로젝트 전체 워크플로우에서 성과를 확인할 수 있는 종합 시각화 시스템을 구축했습니다.

---

## 📊 시각화 구성 요소

### 1. 데이터 생성 단계 (`generate_data.py`)
- ✅ 궤적 샘플 시각화 (6개 랜덤 샘플)
- ✅ 특징 분포 시각화 (jerk, pitch, settling_time)

**사용법:**
```bash
python scripts/generate_data.py --num-episodes 1000 --dataset-id A --visualize
```

### 2. 선호도 데이터셋 구축 (`build_preference_dataset.py`)
- ✅ 클러스터링 결과 시각화 (PCA 투영, 클러스터 크기)
- ✅ 선호도 쌍 분포 시각화

**사용법:**
```bash
python scripts/build_preference_dataset.py \
    --input_path artifacts/A/datasets/1000.pkl \
    --output_path datasets/preference_dataset_A.pkl \
    --visualize
```

### 3. 모델 학습 (`train_model.py`)
- ✅ 학습 곡선 로컬 저장 (Loss, Accuracy, KL Divergence, KL Weight)
- ✅ WandB 의존성 없이 로컬 파일로 저장

**사용법:**
```bash
python scripts/train_model.py \
    --dataset_path datasets/preference_dataset_A.pkl \
    --save_training_curves True \
    ...
```

### 4. 적응 단계 (`run_interactive_adaptation.py`)
- ✅ z 벡터 진화 시각화
- ✅ 적응 요약 시각화 (PCA, context 크기, 변화율)

**사용법:**
```bash
python scripts/run_interactive_adaptation.py \
    --vae_model_path "logs/.../best_model.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/1000.pkl" \
    --visualize
```

### 5. 평가 단계 (`evaluate_adaptation.py`)
- ✅ 보상 분포 시각화
- ✅ 적응 전/후 비교 시각화

**사용법:**
```bash
python scripts/evaluate_adaptation.py \
    --vae_model_path "logs/.../best_model.pt" \
    --adapted_z_path "data/adapted_z.pt" \
    --trajectory_dataset_path "artifacts/A/datasets/1000.pkl" \
    --visualize \
    --before_z_path "data/initial_z.pt"  # optional
```

---

## 📁 생성되는 시각화 파일

```
artifacts/{dataset_id}/visualizations/
  ├─ trajectory_samples.png
  └─ feature_distributions.png

datasets/visualizations/
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

## 🎯 전체 워크플로우 체크리스트

프로젝트 전체 성과 확인을 위한 체크리스트:

1. ✅ **데이터 생성**: `generate_data.py --visualize`
2. ✅ **선호도 데이터셋**: `build_preference_dataset.py --visualize`
3. ✅ **모델 학습**: `train_model.py --save_training_curves True`
4. ✅ **적응 단계**: `run_interactive_adaptation.py --visualize`
5. ✅ **평가 단계**: `evaluate_adaptation.py --visualize`

---

## 📚 관련 파일

- `src/utils/visualization.py`: 모든 시각화 함수 구현
- `VISUALIZATION_GUIDE.md`: 상세 사용 가이드

---

## 💡 주요 특징

1. **단계별 시각화**: 각 워크플로우 단계에서 독립적으로 시각화 생성
2. **로컬 저장**: WandB 없이도 로컬 파일로 결과 확인 가능
3. **종합 비교**: 적응 전/후 성과 비교 가능
4. **프로그래밍 방식**: 함수를 직접 호출하여 커스텀 시각화 가능

이제 프로젝트 전체 성과를 완전히 확인할 수 있습니다!

