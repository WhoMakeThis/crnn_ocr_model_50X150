# CRNN OCR 모델

이 프로젝트는 Convolutional Recurrent Neural Network (CRNN)을 사용한 OCR(광학 문자 인식) 모델입니다.

## 기능
- 이미지에서 텍스트 인식
- 한글 및 영문 지원
- CPU/GPU 학습 지원

## 설치 방법
```bash
pip install -r requirements.txt
```

## 사용 방법
1. 데이터셋 준비
   - `dataset` 폴더에 학습할 이미지 데이터를 준비합니다.

2. 모델 학습
   ```bash
   python train.py
   ```

3. 모델 사용
   ```python
   from crnn.model import CRNN
   model = CRNN.load_from_checkpoint('best_crnn_model.pth')
   ```

## 프로젝트 구조
- `crnn/`: CRNN 모델 관련 코드
- `dataset/`: 학습 데이터셋
- `train.py`: 모델 학습 스크립트
- `requirements.txt`: 필요한 패키지 목록

## 학습된 모델
- `best_crnn_model.pth`: 학습된 최적의 모델 파일

## 라이선스
MIT License

## Files
- `dataset.py` — Custom dataset & collate functions
- `model.py` — CRNN model definition
- `train.py` — Training loop with validation and model saving
- `predict.py` — Inference script with CSV output

## Usage
```bash
python train.py
python predict.py