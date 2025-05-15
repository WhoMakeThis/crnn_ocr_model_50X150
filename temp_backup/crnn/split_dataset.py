import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir='dataset', train_ratio=0.8):
    # 디렉토리 생성
    train_dir = os.path.join(source_dir, 'captcha_images_split/train')
    val_dir = os.path.join(source_dir, 'captcha_images_split/val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    random.shuffle(image_files)
    
    # train/val 분할
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # 파일 복사
    print("학습 데이터셋 복사 중...")
    for f in tqdm(train_files):
        src = os.path.join(source_dir, f)
        dst = os.path.join(train_dir, f)
        shutil.copy2(src, dst)
    
    print("검증 데이터셋 복사 중...")
    for f in tqdm(val_files):
        src = os.path.join(source_dir, f)
        dst = os.path.join(val_dir, f)
        shutil.copy2(src, dst)
    
    print(f"데이터셋 분할 완료:")
    print(f"- 학습 데이터셋: {len(train_files)}개")
    print(f"- 검증 데이터셋: {len(val_files)}개")

if __name__ == "__main__":
    split_dataset() 