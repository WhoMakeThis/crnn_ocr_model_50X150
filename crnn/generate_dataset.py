import os
from PIL import Image, ImageDraw, ImageFont
import random
import string
from tqdm import tqdm

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MAX_LABEL_LEN = 5

def generate_random_text(length=MAX_LABEL_LEN):
    """랜덤 텍스트 생성 (대문자와 숫자 조합)"""
    return ''.join(random.choice(CHARS) for _ in range(length))

def create_text_image(width=150, height=50, text=None):
    """텍스트가 포함된 이미지 생성"""
    if text is None:
        text = generate_random_text()
    
    # 새로운 이미지 생성 (RGB 모드)
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 폰트 설정
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    # 텍스트 위치 계산
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # 텍스트 그리기
    draw.text((x, y), text, font=font, fill='black')
    
    return image, text

def generate_dataset(num_images=50000, output_dir='dataset'):
    """데이터셋 생성"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 생성 및 저장
    for i in tqdm(range(num_images), desc="이미지 생성 중"):
        image, text = create_text_image()
        filename = f"{text}.png"
        image.save(os.path.join(output_dir, filename))
    
    print(f"총 {num_images}개의 이미지가 생성되었습니다.")

if __name__ == "__main__":
    generate_dataset() 