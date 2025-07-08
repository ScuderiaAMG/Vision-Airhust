import cv2 as cv
import os
import time
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# 全局参数
TARGET_FPS = 8
FRAME_INTERVAL = 1.0 / TARGET_FPS
IMAGE_FOLDER = "/home/legion/dataset/file"  # 替换为你的图片路径
NUM_CORES = 10  # 使用 10 核

# 初始化 Haar 级联分类器（每个进程独立加载）
def init_classifier():
    return cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 单进程处理函数
def process_images_parallel(image_paths):
    face_cascade = init_classifier()
    face_count = 0
    start_time = time.time()
    
    while time.time() - start_time < 3600:  # 运行1小时
        for path in image_paths:
            try:
                img = cv.imread(path)
                if img is None:
                    continue
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_count += len(faces)
                time.sleep(max(FRAME_INTERVAL - (time.time() % FRAME_INTERVAL), 0))  # 限速8 FPS
            except Exception as e:
                print(f"ERROR: Processing failed for {path}: {str(e)}")
    return face_count

# 主函数
if __name__ == "__main__":
    image_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.[jJ][pP][gG]")) + \
                 glob.glob(os.path.join(IMAGE_FOLDER, "*.[jJ][pP][eE][gG]")) + \
                 glob.glob(os.path.join(IMAGE_FOLDER, "*.[pP][nN][gG]"))
    
    total_images = len(image_paths)
    print(f"Found {total_images} images in folder: {IMAGE_FOLDER}")

    # 分割任务给多个进程
    chunk_size = max(1, len(image_paths) // NUM_CORES)
    image_chunks = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]

    start_total = time.time()
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(process_images_parallel, chunk) for chunk in image_chunks]
        total_faces = sum(future.result() for future in as_completed(futures))

    duration = time.time() - start_total
    detection_rate = (total_faces / (total_images * (3600 / duration))) * 100  # 计算识别率

    print(f"Total Images Processed: {int(total_images * (3600 / duration))}")
    print(f"Face Detections: {total_faces}")
    print(f"Detection Rate: {detection_rate:.2f}%")
    print(f"Processing Time: {duration:.2f}s")
    print(f"Average FPS: {total_images / duration:.2f}")