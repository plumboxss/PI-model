# 파일 역할 참고 가이드

이 문서는 PI-model 프로젝트의 모든 파일과 디렉토리의 역할을 설명합니다.

---

## 📁 루트 디렉토리 파일

### 문서 및 설정 파일

| 파일 | 역할 | 설명 |
|------|------|------|
| `README.md` | 프로젝트 개요 | 프로젝트 소개, 설치 방법, 워크플로우 설명 |
| `project_intent_description.txt` | 방법론 문서 | 상세한 알고리즘 및 수학적 배경 설명 |
| `CODE_UNDERSTANDING_GUIDE.md` | 코드 이해 가이드 | 코드를 체계적으로 이해하기 위한 학습 가이드 |
| `FILE_REFERENCE.md` | 파일 참고 가이드 | 이 문서 - 모든 파일의 역할 정리 |

### 환경 설정 파일

| 파일 | 역할 | 설명 |
|------|------|------|
| `environment.yml` | Conda 환경 설정 | Conda 가상환경 의존성 정의 (Python 3.8, PyTorch, CUDA 11.3) |
| `requirements.txt` | Pip 의존성 | Python 패키지 의존성 목록 (PyTorch는 수동 설치 필요) |
| `setup_gpu.sh` | GPU 환경 설정 스크립트 | GPU 버전 PyTorch 설치 및 의존성 설치 스크립트 |

---

## 📁 `scripts/` - 실행 스크립트

워크플로우의 각 단계를 실행하는 메인 스크립트들입니다.

### `generate_data.py` ⭐ 1단계
**역할**: 원본 궤적 데이터 생성  
**기능**:
- 다양한 P-제어기 게인으로 차량 시뮬레이션 실행
- 각 궤적의 상태, 행동, 특징 추출
- 오라클로 선호도 평가 (결과는 저장만 하고 실제로는 미사용)
- 병렬 처리로 대량 데이터 생성
- 결과를 pickle 파일로 저장

**입력**: 없음 (시뮬레이션 파라미터는 코드 내부)  
**출력**: `artifacts/{dataset_id}/datasets/{dataset_name}.pkl`

**주요 함수**:
- `run_single_episode()`: 단일 에피소드 시뮬레이션
- `generate_dataset()`: 병렬 처리로 데이터셋 생성

---

### `build_preference_dataset.py` ⭐ 2단계
**역할**: 선호도 데이터셋 구축  
**기능**:
- 원본 궤적 데이터 로드
- 궤적 특징 추출 (jerk, pitch, settling_time)
- K-Means 클러스터링으로 사용자 그룹 생성
- 각 클러스터별 선호도 방향 벡터 정의
- 쌍대 비교 데이터 생성 (궤적 쌍 + 선호도 레이블)

**입력**: `artifacts/{dataset_id}/datasets/{dataset_name}.pkl`  
**출력**: `datasets/preference_dataset_A.pkl`

**핵심 로직**:
- 특징 기반 클러스터링
- 클러스터별 선호도 규칙 적용
- 무작위 궤적 쌍 선택 및 레이블 생성

---

### `train_model.py` ⭐ 3단계
**역할**: VAE 모델 학습  
**기능**:
- 선호도 데이터셋 로드
- VAE 모델 초기화
- 학습 루프 실행 (BCE + KL 손실)
- 코사인 어닐링으로 KL weight 조정
- Weights & Biases로 실험 로깅
- 모델 체크포인트 저장

**입력**: `datasets/preference_dataset_A.pkl`  
**출력**: `logs/{env}/{model_type}/{comment}/s{seed}/model_{epoch}.pt`

**주요 설정**:
- 배치 크기, 학습률, 에폭 수
- 어닐링 스케줄 (cosine, cyclical)
- 조기 종료 (Early Stopping)

---

### `run_interactive_adaptation.py` ⭐ 4단계
**역할**: 상호작용 적응  
**기능**:
- 사전 학습된 VAE 모델 로드
- 궤적 데이터셋 로드
- 고정 비교 세트 C 생성 (메모리 효율적)
- 사용자 입력 기반 적응 루프:
  - 암시적 쌍 찾기 (정보량 최대화)
  - 잠재 벡터 z 재추정
- 최종 적응된 z 벡터 저장

**입력**: 
- VAE 모델 경로
- 궤적 데이터셋 경로

**출력**: `adapted_z.pt`

**핵심 클래스**:
- `AdaptationLoop`: 적응 루프 관리
- `find_implicit_pair()`: 정보량이 많은 궤적 쌍 찾기

---

### `evaluate_adaptation.py` ⭐ 5단계
**역할**: 적응된 보상 함수 평가  
**기능**:
- 사전 학습된 VAE 모델 로드
- 적응된 z 벡터 로드
- 고정된 보상 함수 생성: `r_new(s, a) = r_φ(s, a, z_adapted)`
- 궤적 점수 매기기 (주석 처리됨)
- 후속 RL 학습을 위한 기반 제공

**입력**:
- VAE 모델 경로
- 적응된 z 벡터 경로

**출력**: 보상 함수 객체 (현재는 스켈레톤만 구현)

---

## 📁 `src/` - 핵심 소스 코드

### `src/simulation/` - 시뮬레이션 환경

#### `vehicle_model.py`
**역할**: 차량 물리 모델  
**기능**:
- 차량 동역학 방정식 구현
- Numba JIT 컴파일로 성능 최적화
- 차량 상태 업데이트 (10차원 상태 벡터)
- 전방/후방 서스펜션 모델링

**주요 함수**:
- `vehicle_dynamics()`: 차량 동역학 계산
- `compile_vehicle_model()`: Numba 컴파일

---

#### `plant.py`
**역할**: 시뮬레이션 플랜트 (엔진)  
**기능**:
- 차량 파라미터 관리 (`Vehicle_Parameters`)
- ODE 솔버로 시뮬레이션 실행
- 상태 초기화 및 업데이트
- 설정 파일 로드 (`configs/simulations.yaml`)

**주요 클래스**:
- `Vehicle_Parameters`: 차량 물리 파라미터
- `Plant`: 시뮬레이션 플랜트

---

#### `env.py`
**역할**: Gym 환경 인터페이스  
**기능**:
- OpenAI Gym 호환 환경 구현
- 상태 공간 정의 (10차원)
- 행동 공간 정의 (1차원, [-1, 1])
- `reset()`, `step()` 메서드 제공

**주요 클래스**:
- `SingleScenarioEnv`: 단일 시나리오 환경

---

#### `controller.py`
**역할**: 제어기  
**기능**:
- P-제어기 (Proportional Controller) 구현
- 제어 게인 kp 설정
- 상태 기반 제어 토크 계산

**주요 클래스**:
- `PController`: P-제어기

---

---

#### `bump.py`
**역할**: 도로 장애물 (범프)  
**기능**:
- 도로 프로파일 생성
- 범프 높이 및 위치 정의
- 시뮬레이션에 장애물 추가

---

### `src/models/` - 모델 아키텍처

#### `vae.py` ⭐⭐⭐ 핵심
**역할**: VAE 모델 구현  
**기능**:
- 전체 VAE 아키텍처 정의
- 인코더: 선호도 비교 → 잠재 분포 q(z|D)
- 디코더: (s, a, z) → 보상 r
- 손실 계산: BCE + KL divergence

**주요 클래스**:
- `TrajectoryEncoder`: 궤적 임베딩
- `PairEncoder`: 쌍 인코딩
- `SelfAttentionEncoder`: Transformer 기반 인코더
- `RewardDecoder`: 보상 디코더
- `VAEModel`: 메인 VAE 모델

**핵심 메서드**:
- `encode()`: 선호도 비교 → (μ, Σ)
- `decode()`: (obs, act, z) → reward
- `forward()`: 전체 손실 계산
- `reparameterization()`: VAE 리파라미터화 트릭

---

### `src/data/` - 데이터 처리

#### `loader.py`
**역할**: 데이터 로딩 및 전처리  
**기능**:
- 선호도 데이터셋 로드 (pickle)
- 학습/테스트 분할 (80/20)
- PyTorch Dataset 래퍼
- DataLoader 생성

**주요 클래스**:
- `PreferenceDataset`: PyTorch Dataset 구현

**주요 함수**:
- `get_datasets()`: 데이터셋 및 로더 생성

---

### `src/utils/` - 유틸리티

#### `simulation_utils.py`
**역할**: 시뮬레이션 유틸리티  
**기능**:
- 시뮬레이션 기록 및 저장
- 궤적 특징 추출 (jerk, pitch, settling_time)
- 명령줄 인자 파싱

**주요 클래스**:
- `SimulationRecorder`: 시뮬레이션 결과 기록

**주요 함수**:
- `get_trajectory_features()`: 궤적 특징 계산
- `generate_data_parser()`: 인자 파서 생성

---

#### `training_utils.py`
**역할**: 학습 유틸리티  
**기능**:
- Weights & Biases 로거
- 조기 종료 (Early Stopping)
- 코사인 어닐링 스케줄
- 실험 설정 관리

**주요 클래스**:
- `WandBLogger`: wandb 로깅 래퍼
- `EarlyStopper`: 조기 종료 로직
- `Annealer`: 어닐링 스케줄 (cosine, linear, logistic)

**주요 함수**:
- `define_flags_with_default()`: 설정 플래그 정의
- `set_seed()`: 랜덤 시드 설정

---

#### `plot_utils.py`
**역할**: 시각화 유틸리티  
**기능**:
- 학습 결과 시각화
- VAE 결과 플롯
- 후처리 업데이트

**주요 클래스**:
- `AnnealedLinearSchedule`: 선형 어닐링 스케줄 (미사용)

**주요 함수**:
- `plot_vae()`: VAE 결과 시각화
- `update_posterior()`: 후처리 업데이트

---

## 📁 `configs/` - 설정 파일

### `simulations.yaml`
**역할**: 시뮬레이션 설정  
**내용**:
- 차량 파라미터
- 시뮬레이션 시간
- P-제어기 게인 범위

---

---

## 📁 데이터 디렉토리

### `artifacts/`
**역할**: 원본 궤적 데이터 저장  
**구조**:
```
artifacts/
  {dataset_id}/
    datasets/
      {dataset_name}.pkl
```

---

### `datasets/`
**역할**: 선호도 데이터셋 저장  
**파일**:
- `preference_dataset_A.pkl`: 선호도 비교 데이터셋

---

### `logs/`
**역할**: 학습 결과 및 모델 저장  
**구조**:
```
logs/
  {env}/
    {model_type}/
      {comment}/
        s{seed}/
          model_{epoch}.pt
          best_model.pt
          wandb/
```

---

## 📁 기타 디렉토리

### `venv/`
**역할**: Python 가상환경 (로컬 개발용)

### `temp/`
**역할**: 임시 파일 (wandb 로그 등)

### `__pycache__/`
**역할**: Python 바이트코드 캐시

---

## 🔄 데이터 흐름

```
1. generate_data.py
   → artifacts/{dataset_id}/datasets/{name}.pkl

2. build_preference_dataset.py
   → datasets/preference_dataset_A.pkl

3. train_model.py
   → logs/.../model_{epoch}.pt

4. run_interactive_adaptation.py
   → adapted_z.pt

5. evaluate_adaptation.py
   → 보상 함수 객체
```

---

## ⚠️ 주의사항

1. **메모리 효율성**:
   - `run_interactive_adaptation.py`는 메모리 효율적으로 수정됨
   - 대량 데이터 처리 시 주의

3. **GPU 설정**:
   - `setup_gpu.sh` 또는 시작 스크립트로 GPU 버전 PyTorch 설치 필요

---

## 📊 파일 중요도

### ⭐⭐⭐ 필수 (반드시 이해)
- `src/models/vae.py` - VAE 모델
- `scripts/train_model.py` - 학습 프로세스
- `src/data/loader.py` - 데이터 로딩

### ⭐⭐ 중요
- `scripts/generate_data.py` - 데이터 생성
- `scripts/build_preference_dataset.py` - 선호도 데이터셋
- `src/utils/training_utils.py` - 학습 유틸리티

### ⭐ 참고
- `src/simulation/*` - 시뮬레이션 환경
- `scripts/run_interactive_adaptation.py` - 적응
- `src/utils/plot_utils.py` - 시각화

