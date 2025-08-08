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
from ultralytics import __version__ as ultralytics_version
from packaging import version
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random

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

MIN_IMAGES_PER_CLASS = 800
MAX_IMAGES_PER_CLASS = 1000

test_mode = False
max_test_images = 5  

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
    
    diff_img = np.zeros_like(median_background, dtype=np.float32)
    for img in background_images:
        diff = cv2.absdiff(img, median_background).astype(np.float32)
        diff_img = np.maximum(diff_img, diff)
    
    diff_img = np.clip(diff_img, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    min_area = IMG_SIZE * IMG_SIZE * 0.01  
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            x = max(0, min(x, IMG_SIZE - 10))
            y = max(0, min(y, IMG_SIZE - 10))
            w = max(10, min(w, IMG_SIZE - x))
            h = max(10, min(h, IMG_SIZE - y))
            rois.append((x, y, x+w, y+h))
    
    if not rois:
        rois.append((0, 0, IMG_SIZE, IMG_SIZE))
        print("Warning: No ROIs detected, using full image as ROI")
    
    roi_vis = median_background.copy()
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("background_rois.jpg", roi_vis)
    
    print(f"Detected {len(rois)} regions of interest in background")
    return background_images, rois

background_images, rois = analyze_background()

def visualize_roi_and_background():
    print("Visualizing ROI and background...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    median_img = cv2.imread("median_background.jpg")
    median_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(median_img)
    axes[0].set_title("Median Background")
    axes[0].axis('off')
    
    roi_img = cv2.imread("background_rois.jpg")
    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    axes[1].imshow(roi_img)
    axes[1].set_title(f"Detected ROIs: {len(rois)}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("background_analysis_visualization.jpg")
    plt.close()
    print("Background visualization saved as background_analysis_visualization.jpg")

visualize_roi_and_background()

def place_target_in_roi(background, target, roi):
    x1, y1, x2, y2 = roi
    roi_width = x2 - x1
    roi_height = y2 - y1
    
    if target.shape[2] == 3:
        alpha_channel = np.ones((target.shape[0], target.shape[1]), dtype=target.dtype) * 255
        target = np.dstack((target, alpha_channel))
    
    scale_factor = min(
        roi_width / target.shape[1], 
        roi_height / target.shape[0]
    ) * np.random.uniform(0.5, 0.9)  

    new_width = max(10, int(target.shape[1] * scale_factor))
    new_height = max(10, int(target.shape[0] * scale_factor))
    
    resized_target = cv2.resize(target, (new_width, new_height))
    
    max_x = max(1, roi_width - new_width)
    max_y = max(1, roi_height - new_height)
    pos_x = x1 + np.random.randint(0, max_x)
    pos_y = y1 + np.random.randint(0, max_y)
    
    alpha = resized_target[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    alpha = np.repeat(alpha, 3, axis=2)
    
    bg_region = background[pos_y:pos_y+new_height, pos_x:pos_x+new_width]
    blended = resized_target[:, :, :3] * alpha + bg_region * (1 - alpha)
    background[pos_y:pos_y+new_height, pos_x:pos_x+new_width] = blended
    
    xmin = max(0, pos_x)
    ymin = max(0, pos_y)
    xmax = min(background.shape[1], pos_x + new_width)
    ymax = min(background.shape[0], pos_y + new_height)
    
    return background, (xmin, ymin, xmax, ymax)

def visualize_target_placement():
    print("Visualizing target placement...")
    
    bg_img = background_images[0].copy()
    
    class_name = random.choice(os.listdir(RAW_DATA_DIR))
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not img_files:
        print(f"No images found for class {class_name}")
        return
    
    img_file = random.choice(img_files)
    img_path = os.path.join(class_dir, img_file)
    target_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if target_img is None:
        print(f"Failed to read image: {img_path}")
        return
    
    roi = rois[random.randint(0, len(rois)-1)]
    
    composite_img, bbox = place_target_in_roi(bg_img, target_img, roi)
    
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(composite_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    
    cv2.imwrite("target_placement_visualization.jpg", composite_img)
    print("Target placement visualization saved as target_placement_visualization.jpg")

visualize_target_placement()

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
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))

def process_image(class_name, img_file, aug_idx):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    aug_class_dir = os.path.join(AUG_DATA_DIR, class_name)
    os.makedirs(aug_class_dir, exist_ok=True)
    
    img_path = os.path.join(class_dir, img_file)
    target_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if target_img is None:
        print(f"Failed to read image: {img_path}")
        return
    
    if target_img.shape[2] == 1:  
        target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGRA)
    elif target_img.shape[2] == 3:  
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2BGRA)
    
    bg_idx = np.random.randint(0, len(background_images))
    bg_img = background_images[bg_idx].copy()
    
    roi = rois[np.random.randint(0, len(rois))]
    
    composite_img, bbox = place_target_in_roi(bg_img, target_img, roi)
    
    xmin, ymin, xmax, ymax = bbox
    if xmax - xmin < 5 or ymax - ymin < 5:
        print(f"Invalid bbox size: {bbox} for {img_path}")
        return
    
    try:
        transformed = transform(
            image=composite_img,
            bboxes=[bbox],
            class_labels=[class_name]
        )
    except Exception as e:
        print(f"Transform error: {e}")
        return
    
    aug_img = transformed['image']
    aug_bboxes = transformed['bboxes']
    
    if not aug_bboxes:
        print(f"No bbox after transform for {img_path}")
        return
        
    xmin, ymin, xmax, ymax = aug_bboxes[0]
    xmin = max(0, min(xmin, IMG_SIZE-1))
    ymin = max(0, min(ymin, IMG_SIZE-1))
    xmax = max(1, min(xmax, IMG_SIZE))
    ymax = max(1, min(ymax, IMG_SIZE))
    
    if xmax <= xmin or ymax <= ymin:
        print(f"Invalid transformed bbox: {xmin},{ymin},{xmax},{ymax}")
        return
    
    aug_img_path = os.path.join(aug_class_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_idx}.jpg")
    aug_xml_path = os.path.join(aug_class_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_idx}.xml")
    
    aug_img_np = aug_img.permute(1, 2, 0).numpy()
    aug_img_np = cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(aug_img_path, (aug_img_np * 255).astype(np.uint8))
    
    aug_root = ET.Element("annotation")
    ET.SubElement(aug_root, "folder").text = class_name
    ET.SubElement(aug_root, "filename").text = os.path.basename(aug_img_path)
    
    size = ET.SubElement(aug_root, "size")
    ET.SubElement(size, "width").text = str(IMG_SIZE)
    ET.SubElement(size, "height").text = str(IMG_SIZE)
    ET.SubElement(size, "depth").text = "3"
    
    obj = ET.SubElement(aug_root, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(int(xmin))
    ET.SubElement(bndbox, "ymin").text = str(int(ymin))
    ET.SubElement(bndbox, "xmax").text = str(int(xmax))
    ET.SubElement(bndbox, "ymax").text = str(int(ymax))
    
    tree = ET.ElementTree(aug_root)
    tree.write(aug_xml_path)

print("Starting target placement and labeling (using CPU threads)...")

class_stats = {}
for class_name in os.listdir(RAW_DATA_DIR):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    if os.path.isdir(class_dir):
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_files:
            class_stats[class_name] = len(img_files)

print("\nClass statistics:")
for class_name, count in class_stats.items():
    print(f"- {class_name}: {count} images")

aug_per_class = {}
for class_name, count in class_stats.items():
    if test_mode:
        target_count = min(max_test_images, MIN_IMAGES_PER_CLASS)
    else:
        target_count = random.randint(MIN_IMAGES_PER_CLASS, MAX_IMAGES_PER_CLASS)
    
    if count > 0:
        aug_per_image = max(1, int(target_count / count))
        aug_per_class[class_name] = aug_per_image
        print(f"{class_name}: {count} images -> generating {count * aug_per_image} augmented images (target: {target_count})")

with concurrent.futures.ThreadPoolExecutor(max_workers=config["CPU_CORES"]) as executor:
    futures = []
    total_tasks = 0
    
    for class_name in class_stats.keys():
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        aug_count = aug_per_class.get(class_name, 1)
        
        for img_file in img_files:
            for aug_idx in range(aug_count):
                futures.append(executor.submit(process_image, class_name, img_file, aug_idx))
                total_tasks += 1
    
    progress_bar = tqdm(total=total_tasks, desc="Generating augmented data")
    for future in concurrent.futures.as_completed(futures):
        progress_bar.update(1)
    progress_bar.close()

print(f"Target placement completed. Generated {total_tasks} augmented images.")

print("Converting to YOLO format...")

def convert_to_yolo_format(data_dir, output_dir):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    total_images = 0
    total_labels = 0
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, file)
                    xml_path = os.path.splitext(img_path)[0] + ".xml"
                    
                    if not os.path.exists(xml_path):
                        print(f"XML not found for {img_path}")
                        continue
                    
                    shutil.copy(img_path, os.path.join(images_dir, file))
                    total_images += 1
                    
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        
                        label_path = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")
                        
                        with open(label_path, 'w') as f:
                            for obj in root.findall('object'):
                                cls_name = obj.find('name').text
                                if cls_name not in class_to_id:
                                    print(f"Class {cls_name} not in class list")
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
                                
                                x_center = max(0.0, min(1.0, x_center))
                                y_center = max(0.0, min(1.0, y_center))
                                bbox_width = max(0.01, min(1.0, bbox_width))
                                bbox_height = max(0.01, min(1.0, bbox_height))
                                
                                f.write(f"{class_to_id[cls_name]} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                                total_labels += 1
                    except Exception as e:
                        print(f"Error processing {xml_path}: {e}")
    
    print(f"Converted {total_images} images and {total_labels} labels to YOLO format")
    
    return class_names

class_names = convert_to_yolo_format(AUG_DATA_DIR, TRAIN_DIR)

def visualize_yolo_labels():
    print("Visualizing YOLO labels...")
    
    images_dir = os.path.join(TRAIN_DIR, "images")
    labels_dir = os.path.join(TRAIN_DIR, "labels")
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("No images found for visualization")
        return
    
    img_file = random.choice(image_files)
    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
    
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            x = int((x_center - width/2) * img_w)
            y = int((y_center - height/2) * img_h)
            w = int(width * img_w)
            h = int(height * img_h)
            
            x = max(0, min(x, img_w-1))
            y = max(0, min(y, img_h-1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"Class {class_id}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("YOLO Label Visualization")
    plt.axis('off')
    plt.savefig("yolo_label_visualization.jpg")
    plt.close()
    print("YOLO label visualization saved as yolo_label_visualization.jpg")

visualize_yolo_labels()

with open(os.path.join(YOLO_DATA_DIR, "dataset.yaml"), "w") as f:
    f.write(f"train: {os.path.abspath(TRAIN_DIR)}\n")
    f.write(f"val: {os.path.abspath(TRAIN_DIR)}\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {[str(name) for name in class_names]}\n")

print(f"YOLO dataset prepared with {len(class_names)} classes. Starting training...")

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading YOLOv8x model...")
    
    try:
        model = YOLO("yolov8x.yaml")
        model = model.load("yolov8x.pt")
        print("YOLOv8x model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    model.to(device)

    train_params = {
        'data': os.path.join(YOLO_DATA_DIR, "dataset.yaml"),
        'epochs': 1000,
        'batch': BATCH_SIZE,
        'imgsz': IMG_SIZE,
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
        'amp': True,
        'half': True,
        'plots': True
    }
    
    if version.parse(ultralytics_version) < version.parse("8.0.0"):
        train_params['obj'] = OBJ_GAIN
        train_params['label_smoothing'] = 0.1
    else:
        train_params['kobj'] = OBJ_GAIN
        if 'label_smoothing' in train_params:
            train_params.pop('label_smoothing')
        train_params['dfl'] = 1.0
        train_params['close_mosaic'] = 10
    
    start_time = time.time()
    last_checkpoint_time = start_time
    
    while time.time() - start_time < TRAIN_HOURS * 3600:
        try:
            results = model.train(**dict(train_params, epochs=1))
            
            if results:
                print(f"Epoch results: mAP50={results.results_dict.get('metrics/mAP50', 0):.4f}, "
                      f"precision={results.results_dict.get('metrics/precision', 0):.4f}, "
                      f"recall={results.results_dict.get('metrics/recall', 0):.4f}")
        
        except Exception as e:
            print(f"Training error: {e}")
            break
            
        if time.time() - last_checkpoint_time >= CHECKPOINT_INTERVAL:
            checkpoint_time = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_time}"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            
            weights_dir = os.path.join(checkpoint_path, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            
            try:
                model.save(os.path.join(weights_dir, "last.pt"))
                print(f"\nCheckpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
            
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
    
    try:
        model.export(format="pt")
        
        training_dir = os.path.join(CHECKPOINT_DIR, "yolov8x_training")
        best_model_path = os.path.join(training_dir, "weights", "best.pt")
        
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, final_path)
            print(f"Final model saved to: {final_path}")
        else:
            last_model_path = os.path.join(training_dir, "weights", "last.pt")
            if os.path.exists(last_model_path):
                shutil.copy(last_model_path, final_path)
                print(f"Using last.pt as final model: {final_path}")
            else:
                print("Error: No trained model found to save as final!")
                final_path = None
    except Exception as e:
        print(f"Error saving final model: {e}")
        final_path = None
    
    total_time = (time.time() - start_time) / 3600
    print(f"Training completed. Total training time: {total_time:.2f} hours")
    
    return final_path

if not test_mode:
    final_model_path = train_model()
else:
    print("Test mode complete. Set test_mode=False for full training.")