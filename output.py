import cv2
import os
import numpy as np
import random
from pathlib import Path

input_dir = "/home/legion/dataset/file"
output_dir = input_dir
target_count = 1000

def augment_image(image):
    augmented = image.copy()
    
    # 随机旋转
    if random.random() > 0.5:
        angle = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((augmented.shape[1]//2, augmented.shape[0]//2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (augmented.shape[1], augmented.shape[0]))
    
    # 随机水平翻转
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)
    
    # 随机亮度调整
    if random.random() > 0.5:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.5, 1.5), 0, 255)
        augmented = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    return augmented

def main():
    image_files = [f for f in Path(input_dir).iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    original_count = len(image_files)
    print(f"Found {original_count} original images")
    
    per_image = (target_count - original_count) // original_count + 1
    
    for img_path in image_files:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 保存原始图片
            cv2.imwrite(str(img_path), image)
            
            # 生成增强版本
            for i in range(per_image):
                augmented = augment_image(image)
                new_name = f"{img_path.stem}_aug_{i:03d}{img_path.suffix}"
                new_path = os.path.join(output_dir, new_name)
                cv2.imwrite(new_path, augmented)
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Augmentation complete. Total images: {len(os.listdir(output_dir))}")

if __name__ == "__main__":
    main()