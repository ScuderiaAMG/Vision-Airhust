import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import time
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from ultralytics import YOLO
import pynvml
import psutil
import concurrent.futures
import math

TRAIN_HOURS = 96
CHECKPOINT_INTERVAL = 4 * 3600
BATCH_SIZE = 8
IMG_SIZE = 640
NUM_WORKERS = 12
PIN_MEMORY = True
LEARNING_RATE = 0.001
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
BOX_GAIN = 0.05
CLS_GAIN = 0.5
OBJ_GAIN = 1.0
NUM_BACKGROUND_IMAGES = 17
BACKGROUND_SCENE_DIR = "/home/legion/dataset/background_scene"
MAX_VRAM_USAGE = 6.5

def get_hardware_config():
    config = {
        "GPU_NAME": "Unknown",
        "GPU_MEMORY": 0,
        "CPU_CORES": os.cpu_count() or 4
    }
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        config["GPU_NAME"] = pynvml.nvmlDeviceGetName(handle)
        config["GPU_MEMORY"] = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
    except:
        print("Warning: GPU info unavailable")
    
    print(f"\nHardware configuration:")
    print(f"GPU: {config['GPU_NAME']} | VRAM: {config['GPU_MEMORY']:.1f}GB")
    print(f"CPU: {config['CPU_CORES']} cores")
    
    if config['GPU_MEMORY'] < MAX_VRAM_USAGE + 1:
        print(f"Warning: VRAM usage ({config['GPU_MEMORY']:.1f}GB) close to limit ({MAX_VRAM_USAGE}GB)")
    
    return config

config = get_hardware_config()

RAW_DATA_DIR = "/home/legion/dataset/raw_dataset"
AUG_DATA_DIR = "/home/legion/dataset/augmented_dataset"
YOLO_DATA_DIR = "/home/legion/dataset/yolo_dataset"
TRAIN_DIR = os.path.join(YOLO_DATA_DIR, "train")
VAL_DIR = os.path.join(YOLO_DATA_DIR, "val")
CHECKPOINT_DIR = "/home/legion/dataset"

os.makedirs(AUG_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BACKGROUND_SCENE_DIR, exist_ok=True)

print(f"\nDirectory structure:")
print(f"- Raw data: {os.path.abspath(RAW_DATA_DIR)}")
print(f"- Backgrounds: {os.path.abspath(BACKGROUND_SCENE_DIR)}")
print(f"- Augmented data: {os.path.abspath(AUG_DATA_DIR)}")
print(f"- YOLO dataset: {os.path.abspath(YOLO_DATA_DIR)}")
print(f"- Checkpoints: {os.path.abspath(CHECKPOINT_DIR)}")

def analyze_background():
    print("Analyzing background scenes and detecting ROIs...")
    background_images = []
    valid_files = [f for f in os.listdir(BACKGROUND_SCENE_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not valid_files:
        print("Error: No background images found in background_scene directory")
        exit()
    
    for img_file in tqdm(valid_files[:NUM_BACKGROUND_IMAGES], desc="Loading backgrounds"):
        img_path = os.path.join(BACKGROUND_SCENE_DIR, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            background_images.append(img)
    
    median_background = np.median(background_images, axis=0).astype(np.uint8)
    cv2.imwrite("median_background.jpg", median_background)
    
    foreground_masks = []
    for img in tqdm(background_images, desc="Processing backgrounds"):
        diff = cv2.absdiff(img, median_background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        foreground_masks.append(thresh)
    
    combined_mask = np.max(foreground_masks, axis=0)
    combined_mask = cv2.erode(combined_mask, None, iterations=1)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            rois.append((x, y, x+w, y+h))
    
    roi_vis = median_background.copy()
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("background_rois.jpg", roi_vis)
    
    print(f"Detected {len(rois)} regions of interest in background")
    return background_images, rois

background_images, rois = analyze_background()

def place_target_in_roi(background, target, roi):
    x1, y1, x2, y2 = roi
    roi_width = x2 - x1
    roi_height = y2 - y1
    
    scale_factor = min(roi_width / target.shape[1], roi_height / target.shape[0]) * np.random.uniform(0.7, 0.95)
    new_width = int(target.shape[1] * scale_factor)
    new_height = int(target.shape[0] * scale_factor)
    resized_target = cv2.resize(target, (new_width, new_height))
    
    pos_x = x1 + np.random.randint(0, max(1, roi_width - new_width))
    pos_y = y1 + np.random.randint(0, max(1, roi_height - new_height))
    
    for c in range(0, 3):
        background[pos_y:pos_y+new_height, pos_x:pos_x+new_width, c] = \
            resized_target[:, :, c] * (resized_target[:, :, 3]/255.0) + \
            background[pos_y:pos_y+new_height, pos_x:pos_x+new_width, c] * (1.0 - resized_target[:, :, 3]/255.0)
    
    return background, (pos_x, pos_y, pos_x+new_width, pos_y+new_height)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
    A.CLAHE(p=0.4),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.6),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=0.6),
    A.ToFloat(),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def process_image(class_name, img_file, aug_idx):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    aug_class_dir = os.path.join(AUG_DATA_DIR, class_name)
    os.makedirs(aug_class_dir, exist_ok=True)
    
    img_path = os.path.join(class_dir, img_file)
    target_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if target_img is None:
        return
    
    if len(target_img.shape) < 3 or target_img.shape[2] == 1:
        target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGRA)
    elif target_img.shape[2] == 3:
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2BGRA)
    
    bg_idx = np.random.randint(0, len(background_images))
    bg_img = background_images[bg_idx].copy()
    roi = rois[np.random.randint(0, len(rois))] if rois else (0, 0, bg_img.shape[1], bg_img.shape[0])
    
    composite_img, bbox = place_target_in_roi(bg_img, target_img, roi)
    
    try:
        transformed = transform(
            image=composite_img,
            bboxes=[bbox],
            class_labels=[class_name]
        )
    except Exception as e:
        return
    
    aug_img = transformed['image']
    aug_bboxes = transformed['bboxes']
    
    if aug_bboxes:
        aug_img_path = os.path.join(aug_class_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_idx}.jpg")
        aug_xml_path = os.path.join(aug_class_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_idx}.xml")
        
        aug_img_np = aug_img.permute(1, 2, 0).numpy()
        aug_img_np = cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aug_img_path, (aug_img_np * 255).astype(np.uint8))
        
        aug_root = ET.Element("annotation")
        ET.SubElement(aug_root, "folder").text = class_name
        ET.SubElement(aug_root, "filename").text = os.path.basename(aug_img_path)
        
        size = ET.SubElement(aug_root, "size")
        ET.SubElement(size, "width").text = str(aug_img.shape[2])
        ET.SubElement(size, "height").text = str(aug_img.shape[1])
        ET.SubElement(size, "depth").text = str(aug_img.shape[0])
        
        obj = ET.SubElement(aug_root, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin, ymin, xmax, ymax = aug_bboxes[0]
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))
        
        tree = ET.ElementTree(aug_root)
        tree.write(aug_xml_path)

print("Starting target placement and labeling (using CPU threads)...")

with concurrent.futures.ThreadPoolExecutor(max_workers=config["CPU_CORES"]) as executor:
    futures = []
    
    for class_name in os.listdir(RAW_DATA_DIR):
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in img_files:
            for aug_idx in range(50):
                futures.append(executor.submit(process_image, class_name, img_file, aug_idx))
    
    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating augmented data"):
        pass

print("Target placement completed. Converting to YOLO format...")

def convert_to_yolo_format(data_dir, output_dir):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, file)
                    xml_path = os.path.splitext(img_path)[0] + ".xml"
                    
                    if not os.path.exists(xml_path):
                        continue
                    
                    shutil.copy(img_path, os.path.join(images_dir, file))
                    
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    label_path = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")
                    
                    with open(label_path, 'w') as f:
                        for obj in root.findall('object'):
                            cls_name = obj.find('name').text
                            if cls_name not in class_to_id:
                                continue
                                
                            bndbox = obj.find('bndbox')
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            
                            size = root.find('size')
                            width = float(size.find('width').text)
                            height = float(size.find('height').text)
                            
                            x_center = (xmin + xmax) / (2 * width)
                            y_center = (ymin + ymax) / (2 * height)
                            bbox_width = (xmax - xmin) / width
                            bbox_height = (ymax - ymin) / height
                            
                            f.write(f"{class_to_id[cls_name]} {x_center} {y_center} {bbox_width} {bbox_height}\n")

convert_to_yolo_format(AUG_DATA_DIR, TRAIN_DIR)

class_names = sorted(os.listdir(RAW_DATA_DIR))
with open(os.path.join(YOLO_DATA_DIR, "dataset.yaml"), "w") as f:
    f.write(f"train: {os.path.abspath(TRAIN_DIR)}\n")
    f.write(f"val: {os.path.abspath(TRAIN_DIR)}\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {class_names}")

print("YOLO dataset prepared. Starting training...")

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8x.yaml").load("yolov8x.pt")
    model.to(device)
    
    train_params = {
        'data': os.path.join(YOLO_DATA_DIR, "dataset.yaml"),
        'epochs': 1000,
        'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE,
        'rect': True,
        'cos_lr': True,
        'workers': NUM_WORKERS,
        'device': 0,
        'project': CHECKPOINT_DIR,
        'name': 'yolov8x_training',
        'exist_ok': True,
        'optimizer': 'AdamW',
        'lr0': LEARNING_RATE,
        'momentum': MOMENTUM,
        'weight_decay': WEIGHT_DECAY,
        'warmup_epochs': WARMUP_EPOCHS,
        'box': BOX_GAIN,
        'cls': CLS_GAIN,
        'obj': OBJ_GAIN,
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 30.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.1,
        'perspective': 0.0005,
        'flipud': 0.3,
        'fliplr': 0.3,
        'single_cls': False,
        'label_smoothing': 0.1,
        'patience': 200,
        'amp': True,
        'half': True,
        'plots': True
    }
    
    start_time = time.time()
    last_checkpoint_time = start_time
    
    while time.time() - start_time < TRAIN_HOURS * 3600:
        elapsed = time.time() - last_checkpoint_time
        remaining_interval = max(0, CHECKPOINT_INTERVAL - elapsed)
        
        model.train(**dict(train_params, epochs=1))
        
        if time.time() - last_checkpoint_time >= CHECKPOINT_INTERVAL:
            checkpoint_time = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_time}"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            
            model.save(os.path.join(checkpoint_path, "weights", "last.pt"))
            print(f"\nCheckpoint saved: {checkpoint_path}")
            
            last_checkpoint_time = time.time()
            
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mem = mem_info.used / (1024 ** 3)
                print(f"Current VRAM usage: {used_mem:.2f}GB / {MAX_VRAM_USAGE}GB")
                if used_mem > MAX_VRAM_USAGE:
                    print("Warning: VRAM usage exceeds limit!")
            except:
                pass
    
    final_path = "yolov8x_final.pt"
    model.export(format="pt")
    os.rename(os.path.join(CHECKPOINT_DIR, "yolov8x_training/weights/best.pt"), final_path)
    
    total_time = (time.time() - start_time) / 3600
    print(f"Training completed. Final model saved to: {final_path}")
    print(f"Total training time: {total_time:.2f} hours")
    
    return final_path

final_model_path = train_model()