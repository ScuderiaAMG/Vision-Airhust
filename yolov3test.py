import torch
import os
from PIL import Image
import time
import glob
import psutil

# Load YOLOv3 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').eval().cuda()

# Set target processing speed (8 FPS)
TARGET_FPS = 8
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Define GPU utilization limiter
def limit_gpu_usage(target_utilization=0.3):
    torch.cuda.set_per_process_memory_fraction(target_utilization)

# Image processing function
def process_images(image_folder):
    image_paths = glob.glob(os.path.join(image_folder, "*.[jJ][pP][gG]")) + \
                 glob.glob(os.path.join(image_folder, "*.[pP][nN][gG]"))
    
    total_images = len(image_paths)
    detection_count = 0
    start_time = time.time()
    
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            results = model(img, size=640)
            
            # Count face detections
            detection_count += int(0 in results.pred[0][:, -1].unique().tolist())
            
            # Maintain 8 FPS rate
            time.sleep(max(FRAME_INTERVAL - (time.time() % FRAME_INTERVAL), 0))
            
        except Exception as e:
            print(f"ERROR: Failed to process {path}: {str(e)}")
            continue

    end_time = time.time()
    return detection_count, total_images, end_time - start_time

# Main execution
if __name__ == "__main__":
    # Set GPU memory limit
    limit_gpu_usage(0.3)
    
    IMAGE_FOLDER = "/home/legion/dataset/file"
    print(f"Processing images in folder: {IMAGE_FOLDER}")
    
    start_time = time.time()
    detections, total, duration = process_images(IMAGE_FOLDER)
    run_time = time.time() - start_time
    
    print(f"Total Images Processed: {total}")
    print(f"Face Detections: {detections}")
    print(f"Detection Rate: {(detections/total)*100:.2f}%")
    print(f"Processing Time: {duration:.2f}s")
    print(f"Average FPS: {total/duration:.2f}")
    print(f"Total Runtime: {run_time/3600:.2f} hours")
