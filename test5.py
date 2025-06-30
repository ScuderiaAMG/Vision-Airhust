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

# Configuration
TRAIN_HOURS = 12
CHECKPOINT_INTERVAL = 1
BATCH_SIZE = 32
IMG_SIZE = 640
NUM_WORKERS = 8
PIN_MEMORY = True
LEARNING_RATE = 0.001
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
BOX_GAIN = 0.05
CLS_GAIN = 0.5
OBJ_GAIN = 1.0
NUM_BACKGROUND_IMAGES = 10
BACKGROUND_SCENE_DIR = "background_scene"

################################################################## CONFIGURATION #####################################################################
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_hardware_config():
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    
    cpu_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    return { 
        "GPU_MEMORY": gpu_mem,
        "GPU_NAME": gpu_name,
        "CPU_CORES": cpu_cores,
        "LOGICAL_CORES": logical_cores  
    }

    
config = get_hardware_config()

print("\nNVIDIA RTX NOW ARE AVAILABLE\n")

print(f"GPU: {config['GPU_NAME']} | Gpu memory: {config['GPU_MEMORY']:.1f}GB")
print(f"CPU: {config['CPU_CORES']} CPU cores/{config['LOGICAL_CORES']} Logical cores")
print(f"Model: yolov8x")


################################################################ PATH ######################################################################
RAW_DATA_DIR = "raw_dataset"
AUG_DATA_DIR = "augmented_dataset"
YOLO_DATA_DIR = "yolo_dataset"
TRAIN_DIR = os.path.join(YOLO_DATA_DIR, "train")
VAL_DIR = os.path.join(YOLO_DATA_DIR, "val")
CHECKPOINT_DIR = "yolo_checkpoints"
os.makedirs(AUG_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BACKGROUND_SCENE_DIR, exist_ok=True)

# Phase 0: Background fitting and ROI detection
print("Analyzing background scene and detecting ROIs...")
background_images = []
for img_file in os.listdir(BACKGROUND_SCENE_DIR)[:NUM_BACKGROUND_IMAGES]:
    img_path = os.path.join(BACKGROUND_SCENE_DIR, img_file)
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (640, 640))
            background_images.append(img)

if not background_images:
    print("Error: No background images found in background_scene directory")
    exit()

# Compute median background
median_background = np.median(background_images, axis=0).astype(np.uint8)
cv2.imwrite("median_background.jpg", median_background)

# Detect ROIs using background subtraction
foreground_masks = []
for img in background_images:
    diff = cv2.absdiff(img, median_background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    foreground_masks.append(thresh)

combined_mask = np.max(foreground_masks, axis=0)
combined_mask = cv2.erode(combined_mask, None, iterations=1)

# Pure OpenCV contour detection (no imutils needed)
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rois = []
for c in contours:
    if cv2.contourArea(c) > 1000:
        x, y, w, h = cv2.boundingRect(c)
        rois.append((x, y, x+w, y+h))

# Save ROI visualization
roi_vis = median_background.copy()
for (x1, y1, x2, y2) in rois:
    cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite("background_rois.jpg", roi_vis)

print(f"Detected {len(rois)} regions of interest in background")

# Phase 1: Target placement and auto-labeling
print("Starting target placement and labeling process...")

def place_target_in_roi(background, target, roi):
    x1, y1, x2, y2 = roi
    roi_width = x2 - x1
    roi_height = y2 - y1
    
    # Resize target to fit ROI
    scale_factor = min(roi_width / target.shape[1], roi_height / target.shape[0]) * np.random.uniform(0.7, 0.95)
    new_width = int(target.shape[1] * scale_factor)
    new_height = int(target.shape[0] * scale_factor)
    resized_target = cv2.resize(target, (new_width, new_height))
    
    # Random position within ROI
    pos_x = x1 + np.random.randint(0, max(1, roi_width - new_width))
    pos_y = y1 + np.random.randint(0, max(1, roi_height - new_height))
    
    # Blend target with background
    for c in range(0, 3):
        background[pos_y:pos_y+new_height, pos_x:pos_x+new_width, c] = \
            resized_target[:, :, c] * (resized_target[:, :, 3]/255.0) + \
            background[pos_y:pos_y+new_height, pos_x:pos_x+new_width, c] * (1.0 - resized_target[:, :, 3]/255.0)
    
    return background, (pos_x, pos_y, pos_x+new_width, pos_y+new_height)

transform = A.Compose([
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.8),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.9),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
    A.RandomGamma(gamma_limit=(60, 140), p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.CLAHE(p=0.6),
    A.RandomShadow(p=0.4),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0.5, p=0.3),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.3),
    A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.8),
    A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=50, val_shift_limit=50, p=0.8),
    A.Cutout(num_holes=12, max_h_size=32, max_w_size=32, p=0.7),
    A.Perspective(scale=(0.05, 0.1), p=0.6),
    A.PiecewiseAffine(scale=(0.03, 0.05), p=0.5),
    A.GridDistortion(distort_limit=0.3, p=0.5),
    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.1, p=0.5),
    A.ChannelShuffle(p=0.4),
    A.InvertImg(p=0.2),
    A.ToGray(p=0.3),
    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.3),
    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.3),
    A.RandomShadow(p=0.3),
    A.RandomToneCurve(scale=0.3, p=0.4),
    A.MedianBlur(blur_limit=7, p=0.3),
    A.MotionBlur(blur_limit=11, p=0.4),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
    A.Solarize(threshold=128, p=0.2),
    A.Posterize(num_bits=4, p=0.3),
    A.Equalize(p=0.5),
    A.FancyPCA(alpha=0.3, p=0.5),
    A.ToFloat(),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

for class_name in os.listdir(RAW_DATA_DIR):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    aug_class_dir = os.path.join(AUG_DATA_DIR, class_name)
    os.makedirs(aug_class_dir, exist_ok=True)
    
    if os.path.isdir(class_dir):
        for img_file in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                target_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                
                if target_img is None:
                    print(f"Warning: Failed to read image {img_path}")
                    continue
                    
                # Add alpha channel if missing
                if len(target_img.shape) < 3:
                    continue
                if target_img.shape[2] == 3:
                    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2BGRA)
                elif target_img.shape[2] == 1:
                    target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGRA)
                
                for aug_idx in range(100):
                    # Select random background from scene images
                    bg_idx = np.random.randint(0, len(background_images))
                    bg_img = background_images[bg_idx].copy()
                    
                    # Select random ROI for target placement
                    if rois:
                        roi_idx = np.random.randint(0, len(rois))
                        roi = rois[roi_idx]
                    else:
                        # Fallback if no ROIs detected
                        roi = (0, 0, bg_img.shape[1], bg_img.shape[0])
                    
                    # Place target in background
                    composite_img, bbox = place_target_in_roi(bg_img, target_img, roi)
                    
                    # Apply augmentations
                    try:
                        transformed = transform(
                            image=composite_img,
                            bboxes=[bbox],
                            class_labels=[class_name]
                        )
                    except Exception as e:
                        print(f"Transform error: {e}")
                        continue
                    
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    
                    if aug_bboxes:
                        aug_img_path = os.path.join(aug_class_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_idx}.jpg")
                        aug_xml_path = os.path.join(aug_class_dir, f"{os.path.splitext(img_file)[0]}_aug_{aug_idx}.xml")
                        
                        # Save augmented image
                        aug_img_np = aug_img.permute(1, 2, 0).numpy()
                        aug_img_np = cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(aug_img_path, (aug_img_np * 255).astype(np.uint8))
                        
                        # Create XML for augmented image
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

print("Target placement and augmentation completed. Converting to YOLO format...")

# Phase 2: Convert to YOLO format
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
                    
                    # Copy image
                    shutil.copy(img_path, os.path.join(images_dir, file))
                    
                    # Parse XML
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    # Create YOLO label file
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
                            
                            # Get image dimensions
                            size = root.find('size')
                            width = float(size.find('width').text)
                            height = float(size.find('height').text)
                            
                            # Convert to YOLO format
                            x_center = (xmin + xmax) / (2 * width)
                            y_center = (ymin + ymax) / (2 * height)
                            bbox_width = (xmax - xmin) / width
                            bbox_height = (ymax - ymin) / height
                            
                            f.write(f"{class_to_id[cls_name]} {x_center} {y_center} {bbox_width} {bbox_height}\n")

# Convert augmented data to YOLO format for training
convert_to_yolo_format(AUG_DATA_DIR, TRAIN_DIR)

# Create dataset.yaml
class_names = sorted(os.listdir(RAW_DATA_DIR))
with open(os.path.join(YOLO_DATA_DIR, "dataset.yaml"), "w") as f:
    f.write(f"train: {os.path.abspath(TRAIN_DIR)}\n")
    f.write(f"val: {os.path.abspath(TRAIN_DIR)}\n")  # Using train for validation in this setup
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {class_names}")

print("YOLO dataset prepared. Starting training...")

# Phase 3: YOLOv8 training
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8x.yaml").load("yolov8x.pt")
model.to(device)

# Training configuration
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
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 45.0,
    'translate': 0.2,
    'scale': 0.8,
    'shear': 0.3,
    'perspective': 0.001,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.2,
    'copy_paste': 0.2,
    'erasing': 0.3,
    'augment': True,
    'single_cls': False,
    'label_smoothing': 0.1,
    'patience': 100,
    'freeze': None,
    'amp': True,
    'fraction': 1.0,
    'multi_scale': True,
    'conf': 0.001,
    'iou': 0.7,
    'max_det': 300,
    'half': True,
    'plots': True
}

# Start training
start_time = time.time()
model.train(**train_params)

# Save final model
final_path = "yolov8x_final.pt"
model.export(format="pt")
os.rename(os.path.join(CHECKPOINT_DIR, "yolov8x_training/weights/best.pt"), final_path)
print(f"Training completed. Final model saved to: {final_path}")
print(f"Total training time: {(time.time()-start_time)/3600:.2f} hours")
