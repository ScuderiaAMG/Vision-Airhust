import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.cuda import amp
import time
import os
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import psutil
import pynvml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import shutil
from concurrent.futures import ThreadPoolExecutor

def unlock_gpu_power():
    os.system("sudo nvidia-smi -pl 55")
    os.system("sudo nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1'")

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_hardware_config():
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    
    cpu_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    safe_batch = max(4, min(int(gpu_mem * 6), 128))
    
    ################################################# CONFIG ####################################################################
    return {
        "TOTAL_TRAIN_HOURS": 96,
        "BATCH_SIZE": safe_batch,
        "GRAD_ACCUM_STEPS": 1,
        "NUM_WORKERS": min(cpu_cores * 4, logical_cores - 4),
        "MIXED_PRECISION": True,
        "USE_CUDNN_BENCHMARK": True,
        "MODEL": "resnet152",
        "GPU_MEMORY": gpu_mem,
        "GPU_NAME": gpu_name,
        "CPU_CORES": cpu_cores,
        "LOGICAL_CORES": logical_cores  
    }

unlock_gpu_power()
config = get_hardware_config()

print("\nNVIDIA RTX NOW ARE AVAILABLE\n")

print(f"GPU: {config['GPU_NAME']} | Gpu memory: {config['GPU_MEMORY']:.1f}GB")
print(f"CPU: {config['CPU_CORES']} CPU cores/{config['LOGICAL_CORES']} Logical cores")
print(f"Batch size: {config['BATCH_SIZE']} | Num workers: {config['NUM_WORKERS']}")
print(f"Model: {config['MODEL']}")

################################################### PATH SETTINGS ################################################################
ORIGINAL_DATASET_ROOT = os.path.expanduser('~/dataset/hubei')
AUGMENTED_DATASET_ROOT = os.path.expanduser('~/dataset/hubei_augmented')
CHECKPOINT_DIR = os.path.expanduser('~/D/checkpoints')
FINAL_MODEL_PATH = os.path.expanduser('~/D/data/final_model.pt')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)

def generate_augmentations(class_dir, output_dir, class_name, num_augmentations=50):
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return
    
    actual_images = len(image_files)
    augmentation_factor = max(10, num_augmentations * (3 / actual_images)**2)
    num_augmentations = int(augmentation_factor)
    
    print(f"Class '{class_name}': {actual_images} originals -> Generating {num_augmentations} augmentations")
    
    transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=45, p=0.8),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.8),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(var_limit=(10.0, 100.0), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.2),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            orig_output_path = os.path.join(class_output_dir, f"orig_{img_file}")
            cv2.imwrite(orig_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            for i in range(num_augmentations):
                augmented = transform(image=image)
                aug_img = augmented["image"]
                
                aug_img_np = aug_img.permute(1, 2, 0).numpy() * 255
                aug_img_np = aug_img_np.astype(np.uint8)
                
                output_path = os.path.join(class_output_dir, f"aug_{i}_{os.path.splitext(img_file)[0]}.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR))
                
        except Exception as e:
            print(f"Processing error {img_path}: {str(e)}")

def generate_augmented_dataset():
    print("Launching smart data augmentation engine...")
    start_time = time.time()
    
    if os.path.exists(AUGMENTED_DATASET_ROOT):
        shutil.rmtree(AUGMENTED_DATASET_ROOT)
    os.makedirs(AUGMENTED_DATASET_ROOT)
    
    classes = os.listdir(ORIGINAL_DATASET_ROOT)
    total_original_images = 0
    
    for cls in classes:
        cls_dir = os.path.join(ORIGINAL_DATASET_ROOT, cls)
        if os.path.isdir(cls_dir):
            total_original_images += len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Original dataset: {len(classes)} classes, {total_original_images} images")
    
    with ThreadPoolExecutor(max_workers=config['NUM_WORKERS']) as executor:
        futures = []
        for class_name in classes:
            class_dir = os.path.join(ORIGINAL_DATASET_ROOT, class_name)
            if os.path.isdir(class_dir):
                futures.append(executor.submit(generate_augmentations, class_dir, AUGMENTED_DATASET_ROOT, class_name))
        
        for future in futures:
            future.result()
    
    augmented_classes = os.listdir(AUGMENTED_DATASET_ROOT)
    total_augmented_images = 0
    
    for cls in augmented_classes:
        cls_dir = os.path.join(AUGMENTED_DATASET_ROOT, cls)
        if os.path.isdir(cls_dir):
            total_augmented_images += len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Augmented dataset: {len(augmented_classes)} classes, {total_augmented_images} images")
    print(f"Augmentation completed! Time: {(time.time()-start_time)/60:.2f} minutes")
    print(f"Total data increase: {total_augmented_images/total_original_images:.1f}x")

class UltraFastDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, file), self.class_to_idx[class_name]))
        
        self.buffer = np.zeros((len(self.samples), 224, 224, 3), dtype=np.uint8)
        self.labels = np.zeros(len(self.samples), dtype=np.int64)
        
        print(f"Preloaded {len(self.samples)} pictures to RAM...")
        with ThreadPoolExecutor(max_workers=config['NUM_WORKERS']) as executor:
            futures = []
            for idx, (img_path, label) in enumerate(self.samples):
                futures.append(executor.submit(self._load_image, idx, img_path, label))
            
            for future in futures:
                idx, label = future.result()
                self.labels[idx] = label
        
        print("Dataset Preloading Mission Finished.")
    
    def _load_image(self, idx, img_path, label):
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            self.buffer[idx] = img
            return idx, label
        except Exception as e:
            print(f"Loading error! {img_path}: {str(e)}")
            return idx, label
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = self.buffer[idx]
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        
        return img, label

def create_power_model(num_classes):
    if config['MODEL'] == "resnet18":
        model = torchvision.models.resnet152(weights='IMAGENET1K_V1')
    elif config['MODEL'] == "efficientnet_b7":
        model = torchvision.models.efficientnet_b7(weights='IMAGENET1K_V1')
    else:
        model = torchvision.models.resnet101(weights='IMAGENET1K_V1')
    
    if hasattr(model, 'fc'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'classifier'):
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    for param in model.parameters():
        param.requires_grad = True
    
    return model

class GPUMonitor:
    def __init__(self):
        self.handle = handle
        self.power_samples = []
        self.util_samples = []
    
    def get_metrics(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        
        self.power_samples.append(power)
        self.util_samples.append(util)
        
        if len(self.power_samples) > 100:
            self.power_samples.pop(0)
            self.util_samples.pop(0)
        
        return {
            "util": util,
            "mem_used": mem_info.used / (1024**3),
            "mem_total": mem_info.total / (1024**3),
            "power": power,
            "avg_power": sum(self.power_samples) / len(self.power_samples),
            "avg_util": sum(self.util_samples) / len(self.util_samples)
        }

def train(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    monitor = GPUMonitor()
    
    train_transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    train_set = UltraFastDataset(
        root_dir=AUGMENTED_DATASET_ROOT,
        transform=train_transform
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config['BATCH_SIZE'] // world_size,
        sampler=sampler,
        num_workers=min(4, config['NUM_WORKERS']) // world_size,
        # num_workers=config['NUM_WORKERS'] // world_size,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    model = create_power_model(len(train_set.classes))
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=int(config['TOTAL_TRAIN_HOURS'] * 3600 / (len(train_loader) * (config['BATCH_SIZE'] // world_size))),
        pct_start=0.3
    )
    
    scaler = amp.GradScaler(enabled=config['MIXED_PRECISION'])
    
    start_time = time.time()
    last_print = start_time
    last_checkpoint = start_time
    
    print(f"[GPU {rank}] GPU RTX Available now, loading...")
    warmup_data = torch.rand(128, 3, 224, 224, device=device)
    for _ in range(100):
        with amp.autocast(enabled=config['MIXED_PRECISION']):
            _ = model(warmup_data)
    torch.cuda.synchronize()
    print(f"[GPU {rank}] RTX loading mission finished...")

    accumulation_steps = config['GRAD_ACCUM_STEPS']
    
    model.train()
    for epoch in range(10000):
        sampler.set_epoch(epoch)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with amp.autocast(enabled=config['MIXED_PRECISION']):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss / accumulation_steps).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            current_time = time.time()
            if current_time - last_print > 5:
                metrics = monitor.get_metrics()
                lr = optimizer.param_groups[0]['lr']
                
                print(f"[GPU {rank}] Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                      f"GPU: {metrics['util']}% | Power: {metrics['power']:.1f}W | "
                      f"Mem: {metrics['mem_used']:.1f}/{metrics['mem_total']:.1f}GB")
                
                last_print = current_time
            
            if current_time - last_checkpoint > 3600:
                elapsed_hours = (current_time - start_time) / 3600
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{int(elapsed_hours)}h_rank{rank}.pt")
                
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'batch': batch_idx,
                    'elapsed_hours': elapsed_hours
                }, checkpoint_path)
                
                print(f"[GPU {rank}] Checkpoints saved: {checkpoint_path}")
                last_checkpoint = current_time
            
            if current_time - start_time > config['TOTAL_TRAIN_HOURS'] * 3600:
                print(f"[GPU {rank}] Training completed!")
                dist.destroy_process_group()
                return

def main():
    torch.backends.cudnn.benchmark = True
    
    if not os.path.exists(AUGMENTED_DATASET_ROOT) or not os.listdir(AUGMENTED_DATASET_ROOT):
        generate_augmented_dataset()
    else:
        print("Using existing augmented dataset")
    
    world_size = torch.cuda.device_count()
    print(f"{world_size} CUDA devices available")
    
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 1)
    
    model = create_power_model(10)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved: {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
