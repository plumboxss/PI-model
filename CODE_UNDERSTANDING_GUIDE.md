# 코드 이해 가이드

이 문서는 PI-model 프로젝트의 코드를 체계적으로 이해하기 위한 가이드입니다.

## 📚 이해 순서 (단계별)

### **Phase 1: 개념 이해** (30분)

#### 1.1 핵심 개념 문서 읽기
- `README.md` - 프로젝트 개요와 워크플로우
- `project_intent_description.txt` - 상세 방법론 (선택적, 깊이 이해하고 싶을 때)

**핵심 개념:**
- VAE 기반 선호도 학습
- 잠재 벡터 z를 통한 보상 함수 조건화
- 빠른 적응 (Fast Adaptation)

---

### **Phase 2: 시뮬레이션 환경** (1시간)

#### 2.1 차량 시뮬레이션 기초
**읽는 순서:**
1. `src/simulation/vehicle_model.py` - 차량 물리 모델
2. `src/simulation/plant.py` - 플랜트 (시뮬레이션 엔진)
3. `src/simulation/env.py` - Gym 환경 인터페이스
4. `src/simulation/controller.py` - P-제어기
5. `configs/simulations.yaml` - 시뮬레이션 설정

**이해 포인트:**
- 차량의 물리적 동역학
- 상태 공간 (10차원)
- 행동 공간 (1차원, 제어 토크)

#### 2.2 시뮬레이션 실행 흐름
- `src/utils/simulation_utils.py` - 시뮬레이션 유틸리티
  - `SimulationRecorder`: 궤적 기록
  - `get_trajectory_features()`: 궤적 특징 추출 (jerk, pitch, settling_time)

---

### **Phase 3: 데이터 생성 및 처리** (1시간)

#### 3.1 원본 궤적 데이터 생성
**파일:** `scripts/generate_data.py`

**흐름:**
1. 다양한 P-게인으로 시뮬레이션 실행
2. 각 궤적의 상태, 행동, 특징 추출
3. 오라클로 선호도 평가
4. pickle 파일로 저장

**핵심 함수:**
- `run_single_episode()`: 단일 에피소드 실행
- `generate_dataset()`: 병렬 처리로 대량 데이터 생성

#### 3.2 선호도 데이터셋 구축
**파일:** `scripts/build_preference_dataset.py`

**흐름:**
1. 원본 궤적 로드
2. K-Means 클러스터링으로 사용자 그룹 생성
3. 각 그룹별 선호도 규칙 정의
4. 쌍대 비교 데이터 생성

**핵심 개념:**
- 특징 기반 클러스터링
- 선호도 방향 벡터
- 쌍대 비교 레이블 생성

#### 3.3 데이터 로더
**파일:** `src/data/loader.py`

**핵심 클래스:**
- `PreferenceDataset`: PyTorch Dataset 래퍼
- `get_datasets()`: 학습/테스트 데이터셋 생성

---

### **Phase 4: 모델 아키텍처** (2시간) ⭐ 핵심

#### 4.1 VAE 모델 구조
**파일:** `src/models/vae.py`

**읽는 순서:**
1. `TrajectoryEncoder` (5-16줄) - 궤적 임베딩
2. `PairEncoder` (18-28줄) - 쌍 인코딩
3. `SelfAttentionEncoder` (30-90줄) - Transformer 기반 인코더
4. `VAEModel` (113-267줄) - 메인 VAE 모델

**VAEModel의 핵심 메서드:**
- `encode()`: 선호도 비교 → 잠재 분포 q(z|D)
- `decode()`: (s, a, z) → 보상 r
- `forward()`: 전체 손실 계산 (BCE + KL)

**이해 포인트:**
- 인코더: 선호도 비교 쌍 → 잠재 벡터 z의 분포
- 디코더: 상태-행동-잠재벡터 → 보상
- VAE 손실: 재구성 손실 + KL divergence

#### 4.2 어닐링 스케줄
**파일:** `src/utils/training_utils.py`
- `Annealer` 클래스: 코사인 어닐링 구현
- KL weight를 점진적으로 증가

---

### **Phase 5: 학습 과정** (1.5시간)

#### 5.1 학습 스크립트
**파일:** `scripts/train_model.py`

**흐름:**
1. 데이터셋 로드
2. VAE 모델 초기화
3. 어닐러 설정
4. 학습 루프:
   - 배치별 forward pass
   - 손실 계산 (BCE + KL)
   - 역전파 및 최적화
   - 어닐러 업데이트

**핵심 함수:**
- `main()`: 전체 학습 프로세스
- `log_metrics()`: wandb 로깅

#### 5.2 학습 유틸리티
**파일:** `src/utils/training_utils.py`
- `WandBLogger`: 실험 로깅
- `EarlyStopper`: 조기 종료
- `Annealer`: 어닐링 스케줄

---

### **Phase 6: 적응 및 평가** (1시간)

#### 6.1 상호작용 적응
**파일:** `scripts/run_interactive_adaptation.py`

**핵심 클래스:**
- `AdaptationLoop`: 적응 루프 관리
- `find_implicit_pair()`: 정보량이 많은 쌍 찾기

**흐름:**
1. 사전 학습된 모델 로드
2. 궤적 데이터셋 로드
3. 비교 세트 C 생성
4. 반복:
   - 사용자 입력 받기
   - 암시적 쌍 찾기
   - z 재추정

#### 6.2 보상 함수 평가
**파일:** `scripts/evaluate_adaptation.py`
- 적응된 z로 보상 함수 생성
- `get_reward_fn()`: 고정된 보상 함수 반환

---

## 🔍 코드 읽기 팁

### 1. 데이터 흐름 추적
```
generate_data.py → build_preference_dataset.py → train_model.py
```

### 2. 모델 호출 체인
```
VAEModel.forward() 
  → encode() → SelfAttentionEncoder
  → decode() → RewardDecoder
  → 손실 계산
```

### 3. 핵심 수식 매핑
- **인코더**: `q_ψ(z|D) = N(μ, Σ)` → `SelfAttentionEncoder`
- **디코더**: `r = r_φ(s, a, z)` → `RewardDecoder`
- **손실**: `L = BCE + β·KL` → `VAEModel.forward()`

### 4. 디버깅 포인트
- `src/models/vae.py:241` - KL weight 계산
- `src/models/vae.py:242` - 최종 손실
- `scripts/train_model.py:178` - forward pass

---

## 📖 파일별 중요도

### ⭐⭐⭐ 필수 (반드시 이해)
- `src/models/vae.py` - VAE 모델
- `scripts/train_model.py` - 학습 프로세스
- `src/data/loader.py` - 데이터 로딩

### ⭐⭐ 중요 (핵심 이해)
- `scripts/generate_data.py` - 데이터 생성
- `scripts/build_preference_dataset.py` - 선호도 데이터셋
- `src/utils/training_utils.py` - 학습 유틸리티

### ⭐ 참고 (필요시)
- `src/simulation/*` - 시뮬레이션 환경
- `scripts/run_interactive_adaptation.py` - 적응
- `src/utils/plot_utils.py` - 시각화

---

## 🎯 체크리스트

각 단계를 완료했는지 확인하세요:

- [ ] Phase 1: 개념 이해 완료
- [ ] Phase 2: 시뮬레이션 환경 이해
- [ ] Phase 3: 데이터 생성 흐름 이해
- [ ] Phase 4: VAE 모델 아키텍처 이해 ⭐
- [ ] Phase 5: 학습 과정 이해
- [ ] Phase 6: 적응 프로세스 이해

---

## 💡 추가 학습 자료

1. **VAE 기초**: Variational Autoencoder 논문
2. **Transformer**: Attention Is All You Need
3. **Preference Learning**: Bradley-Terry 모델
4. **Annealing**: KL Annealing in VAE

---

## 🐛 문제 해결

코드를 이해하다 막히면:
1. 해당 함수의 docstring 확인
2. 호출하는 곳에서 어떻게 사용되는지 확인
3. 변수명과 주석으로 의도 파악
4. 작은 예제로 직접 실행해보기

