import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data 
import DataLoader
import time
import os

TOTAL_TRAIN_HOURS = 96  
CHECKPOINT_INTERVAL = 4  
NUM_CLASSES = 10  
BATCH_SIZE = 64
LEARNING_RATE = 0.001

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Application available: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=2
    )

    model = ImageClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    last_checkpoint = start_time
    total_seconds = TOTAL_TRAIN_HOURS * 3600
    
    print(f"train started,time: {TOTAL_TRAIN_HOURS}hours..")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    epoch = 0
    while time.time() - start_time < total_seconds:
        epoch += 1
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            current_time = time.time()
            elapsed_hours = (current_time - start_time) / 3600
            if current_time - last_checkpoint > 3600:
                print(f"Have been trained: {elapsed_hours:.1f}/{TOTAL_TRAIN_HOURS}hours | "
                      f"Epoch: {epoch} | Loss: {running_loss/(i+1):.4f}")
                last_checkpoint = current_time
        
        current_time = time.time()
        if current_time - last_checkpoint > CHECKPOINT_INTERVAL * 3600:
            checkpoint_path = f"checkpoints/model_{int(elapsed_hours)}h.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint Saved At: {checkpoint_path}")
            last_checkpoint = current_time
    
    final_path = "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Train finished! Path to final results: {final_path}")
    print(f"Total training hours: {(time.time()-start_time)/3600:.2f}hours")

if __name__ == "__main__":
    main()
