import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import CRNN
import os
import time

# 设置设备
device = torch.device('cpu')
print(f'Using device: {device}')

# 超参数
batch_size = 32
num_epochs = 30
learning_rate = 0.001

# 创建数据集和数据加载器
train_dataset = TextDataset('dataset/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建模型
model = CRNN().to(device)
criterion = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
def train():
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
           
            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long)
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, '
              f'Time: {epoch_time:.2f}s')
        
        
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    print("학습을 시작합니다...")
    train()
    print("학습이 완료되었습니다！") 
