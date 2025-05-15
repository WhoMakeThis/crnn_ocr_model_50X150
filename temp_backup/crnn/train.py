import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CaptchaDataset, CHARS, collate_fn
from model import CRNN
from tqdm import tqdm

# 하이퍼파라미터 설정
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = torch.device("cpu")
SAVE_PATH = "best_crnn_model.pth"

# 손실 함수: CTC Loss
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

def train():
    # 데이터셋 로딩
    train_dataset = CaptchaDataset("dataset")
    val_dataset = CaptchaDataset("dataset")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 모델 초기화
    model = CRNN(50, 1, len(CHARS) + 1, 256).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for imgs, labels, label_lengths in loop:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)

            preds = model(imgs)
            preds_log_softmax = preds.log_softmax(2)

            preds_size = torch.IntTensor([preds.size(0)] * preds.size(1)).to(DEVICE)

            loss = criterion(preds_log_softmax, labels, preds_size, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for imgs, labels, label_lengths in loop:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                label_lengths = label_lengths.to(DEVICE)

                preds = model(imgs)
                preds_log_softmax = preds.log_softmax(2)
                preds_size = torch.IntTensor([preds.size(0)] * preds.size(1)).to(DEVICE)

                loss = criterion(preds_log_softmax, labels, preds_size, label_lengths)
                val_loss += loss.item()
                loop.set_postfix(val_loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Best model saved with val loss {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
    print("학습 완료! 예측 및 평가를 시작합니다...")
    os.system("python crnn/predict.py")
    os.system("python crnn/calculate_accuracy.py")
