#!/bin/bash
# GPU 환경 설정 스크립트

# PyTorch GPU 버전 설치 (CUDA 12.1)
# 이미 PyTorch가 설치되어 있어도 최신 버전으로 업데이트
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성 설치
pip install numpy pandas scikit-learn gym==0.26.2 numba absl-py cloudpickle matplotlib PyYAML tqdm wandb ml_collections

# GPU 확인
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "GPU 환경 설정이 완료되었습니다!"

