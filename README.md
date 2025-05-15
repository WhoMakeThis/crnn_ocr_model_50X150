# CRNN OCR 모델


## 프로젝트 구조
- `crnn/`: CRNN 모델 관련 코드
- `dataset/`: 학습 데이터셋
- `train.py`: 모델 학습 스크립트
- `requirements.txt`: 필요한 패키지 목록


## Files
- `dataset.py` — Custom dataset & collate functions
- `model.py` — CRNN model definition
- `train.py` — Training loop with validation and model saving
- `predict.py` — Inference script with CSV output
