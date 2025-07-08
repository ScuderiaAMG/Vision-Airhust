import cv2 as cv
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial

def process_image(path, face_cascade):
    try:
        img = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return len(faces) > 0
    except:
        return False

def main():
    # 初始化Haar分类器
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    IMAGE_FOLDER = "/home/legion/dataset/file"  # Replace with your image folder path
    print(f"Processing images in folder: {IMAGE_FOLDER}")
    
    image_paths = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_images = len(image_paths)
    detection_results = []
    
    start_time = time.time()
    frame_interval = 1.0 / 8  # 8 FPS目标
    
    with Pool(10) as pool:  # 使用10核CPU
        process_func = partial(process_image, face_cascade=face_cascade)
        
        for result in pool.imap_unordered(process_func, image_paths):
            detection_results.append(result)
            time.sleep(max(frame_interval - (time.time() % frame_interval), 0))
    
    end_time = time.time()
    duration = end_time - start_time
    
    valid_results = [r for r in detection_results if r is not None]
    face_count = sum(valid_results)
    accuracy = (face_count / len(valid_results)) * 100 if valid_results else 0
    
    print(f"Total Images Processed: {total_images}")
    print(f"Face Detections: {face_count}")
    print(f"Detection Rate: {accuracy:.2f}%")
    print(f"Processing Time: {duration:.2f}s")
    print(f"Average FPS: {len(valid_results)/duration:.2f}")

if __name__ == "__main__":
    main()