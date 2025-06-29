import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda import amp
import time
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import shutil
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

##################################### configs #####################################

TOTAL_TRAIN_HOURS = 96
CHECKPOINT_INTERVAL = 4  
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 20
MIXED_PRECISION = True
USE_CUDNN_BENCHMARK = True

AUGMENTATION_FACTOR = 20  
BG_REPLACE = True  
BG_DIR = os.path.expanduser('~/backgrounds')  

#####################################  paths #####################################

ORIGINAL_DATASET_ROOT = os.path.expanduser('~/dataset/hubei')  
AUGMENTED_DATASET_ROOT = os.path.expanduser('~/dataset/hubei_augmented')  
CHECKPOINT_DIR = os.path.expanduser('~/D/checkpoints')  
FINAL_MODEL_PATH = os.path.expanduser('~/D/data/final_model.pt')  

os.makedirs(ORIGINAL_DATASET_ROOT, exist_ok=True)
os.makedirs(AUGMENTED_DATASET_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
os.makedirs(BG_DIR, exist_ok=True)

def remove_background(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def apply_random_background(image, mask, backgrounds):
    if not backgrounds:
        return image
    
    bg_path = random.choice(backgrounds)
    try:
        background = cv2.imread(bg_path)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        background = cv2.resize(background, (image.width, image.height))
    except:
        background = np.ones((image.height, image.width, 3), dtype=np.uint8) * 255
    
    fg = np.array(image)
    result = np.where(mask[..., None] == 0, fg, background)
    
    return Image.fromarray(result)

def generate_augmentations(class_dir, output_dir, class_name, backgrounds, num_augmentations):
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    image_files = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        return
    
    transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.8),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
            
            if BG_REPLACE:
                mask = remove_background(image)
                bg_image = apply_random_background(image, mask, backgrounds)
            else:
                bg_image = image.copy()
            
            orig_output_path = os.path.join(output_dir, class_name, f"orig_{img_file}")
            bg_image.save(orig_output_path)
            
            for i in range(num_augmentations):
                img_cv = cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2BGR)
                
                augmented = transform(image=img_cv)
                aug_img = augmented["image"]
                
                aug_img_np = aug_img.permute(1, 2, 0).numpy() * 255
                aug_img_np = aug_img_np.astype(np.uint8)
                aug_img_pil = Image.fromarray(aug_img_np)
                
                output_path = os.path.join(output_dir, class_name, f"aug_{i}_{img_file}")
                aug_img_pil.save(output_path)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def generate_augmented_dataset():
    print("Generating augmented dataset...")
    start_time = time.time()
    backgrounds = []
    if BG_REPLACE and os.path.exists(BG_DIR):
        backgrounds = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    classes = os.listdir(ORIGINAL_DATASET_ROOT)
    for class_name in classes:
        class_dir = os.path.join(ORIGINAL_DATASET_ROOT, class_name)
        if os.path.isdir(class_dir):
            print(f"Augmenting class: {class_name}")
            generate_augmentations(class_dir, AUGMENTED_DATASET_ROOT, 
                                  class_name, backgrounds, AUGMENTATION_FACTOR)
    
    print(f"Augmentation completed in {(time.time()-start_time)/60:.2f} minutes")

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_val=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_val = is_val
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self.targets = []  
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for file in class_files:
                img_path = os.path.join(class_dir, file)
                self.samples.append((img_path, self.class_to_idx[class_name]))
                self.targets.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_model(num_classes):
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model

def tta_predict(model, image, transform, device, n_aug=10):
    model.eval()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0).to(device))
        
        for _ in range(n_aug):
            aug_img = transform(image)
            outputs += model(aug_img.unsqueeze(0).to(device))
        
        return outputs / (n_aug + 1)

def evaluate_with_tta(model, data_loader, train_transform, device):
    model.eval()
    correct = 0
    total = 0
    
    tta_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            batch_correct = 0
            for i in range(inputs.size(0)):
                img = inputs[i].cpu()
                label = labels[i]
                
                output = tta_predict(model, img, tta_transform, device)
                _, predicted = output.max(1)
                
                if predicted.item() == label.item():
                    batch_correct += 1
            
            correct += batch_correct
            total += inputs.size(0)
    
    accuracy = 100 * correct / total
    print(f"TTA verifing accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    if not os.path.exists(AUGMENTED_DATASET_ROOT) or not os.listdir(AUGMENTED_DATASET_ROOT):
        generate_augmented_dataset()
    else:
        print("Using existing augmented dataset")
    
    if torch.cuda.is_available() and USE_CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_set = CustomDataset(
        root_dir=AUGMENTED_DATASET_ROOT,
        transform=train_transform
    )
    
    val_set = CustomDataset(
        root_dir=ORIGINAL_DATASET_ROOT,
        transform=val_transform,
        is_val=True
    )
    
    class_counts = Counter(train_set.targets)
    print("\nCategories:")
    for cls_idx, count in class_counts.items():
        cls_name = train_set.classes[cls_idx]
        print(f"  {cls_name}: {count} pictures")
    
    weights = [1.0 / class_counts[cls] for cls in train_set.targets]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_set),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    NUM_CLASSES = len(train_set.classes)
    print(f"\n{NUM_CLASSES}  classes detected in dataset.")
    model = create_model(NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    scaler = amp.GradScaler(enabled=MIXED_PRECISION)
    
    start_time = time.time()
    last_checkpoint = start_time
    last_print_time = start_time
    last_val_time = start_time
    total_seconds = TOTAL_TRAIN_HOURS * 3600
    
    print(f"\nTraing mission started | Total time: {TOTAL_TRAIN_HOURS}hours...")
    print(f"batch_size: {BATCH_SIZE}, num_works: {NUM_WORKERS}, grad_accum_Steps: {GRAD_ACCUM_STEPS}")
    
    epoch = 0
    global_step = 0
    best_accuracy = 0.0
    best_tta_accuracy = 0.0
    
    fine_tuning_enabled = False
    
    while time.time() - start_time < total_seconds:
        epoch += 1
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        elapsed_hours = (time.time() - start_time) / 3600
        if not fine_tuning_enabled and elapsed_hours > TOTAL_TRAIN_HOURS * 0.8:
            print("\nfine turning available...")
            for name, param in model.named_parameters():
                if 'layer4' in name or 'layer3' in name or 'fc' in name:
                    param.requires_grad = True
            fine_tuning_enabled = True
            
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=LEARNING_RATE/10, 
                weight_decay=1e-4
            )
            print("Optimizer available now.")
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            global_step += 1
            
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with amp.autocast(enabled=MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            
            current_time = time.time()
            if current_time - last_print_time >= 3600:
                elapsed_hours = (current_time - start_time) / 3600
                avg_loss = running_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]['lr']
                
                print(f"Having trained: {elapsed_hours:.1f}/{TOTAL_TRAIN_HOURS}hours | "
                      f"eopch: {epoch} | batch: {batch_idx+1}/{len(train_loader)} | "
                      f"average loss: {avg_loss:.4f} | learning rate: {lr:.2e}")
                
                last_print_time = current_time
                running_loss = 0.0
            
            if current_time - last_checkpoint >= CHECKPOINT_INTERVAL * 3600:
                elapsed_hours = (current_time - start_time) / 3600
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{int(elapsed_hours)}h.pt")
                
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'class_to_idx': train_set.class_to_idx,
                    'loss': loss.item(),
                    'elapsed_hours': elapsed_hours
                }, checkpoint_path)
                
                print(f"Checkpoint has been saved at: {checkpoint_path}")
                last_checkpoint = current_time
        
        if current_time - last_val_time >= 3600:
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(val_loader)
            print(f"Accuracy verified: {accuracy:.2f}% | Loss verified: {avg_val_loss:.4f}")
            
            if epoch % 4 == 0:
                tta_accuracy = evaluate_with_tta(model, val_loader, train_transform, device)
                if tta_accuracy > best_tta_accuracy:
                    best_tta_accuracy = tta_accuracy
                    tta_model_path = os.path.join(CHECKPOINT_DIR, "best_tta_model.pt")
                    torch.save(model.state_dict(), tta_model_path)
                    print(f"New model. Accuracy: {tta_accuracy:.2f}% Saved at: {tta_model_path}")
            
            scheduler.step(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model. Accuracy: {accuracy:.2f}% Saved at: {best_model_path}")
            
            last_val_time = current_time
            model.train()
    
    print("\nFinal TTA evaluation...")
    final_tta_accuracy = evaluate_with_tta(model, val_loader, train_transform, device)
    
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    
    tta_final_path = os.path.join(CHECKPOINT_DIR, "final_tta_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tta_accuracy': final_tta_accuracy,
        'class_to_idx': train_set.class_to_idx
    }, tta_final_path)
    
    total_time = (time.time() - start_time) / 3600
    print(f"\nTraining mission completed.")
    print(f"Total training time: {total_time:.2f}小时")
    print(f"Final model saved at: {FINAL_MODEL_PATH}")
    print(f"Best verified accuracy: {best_accuracy:.2f}%")
    print(f"Final TTA accuracy: {final_tta_accuracy:.2f}%")

if __name__ == "__main__":
    main()