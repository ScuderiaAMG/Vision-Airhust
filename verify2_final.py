import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import time
import numpy as np
from PIL import Image
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = '/home/legion/hubei_test_models/final_model.pt'
IMAGE_DIR = '/home/legion/dataset'
RUN_DURATION = 3600
NUM_CLASSES = 16
CLASS_LABELS = ['camel', 'chair', 'dolphin', 'lion', 'hamster', 'maple', 'orange', 'orchid', 
                'pickuptruck', 'pinetree', 'rabbit', 'skycraper', 'squirrel', 'tractor', 'turtle', 'willow']

def load_model(model_path, num_classes):
    model = resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        state_dict = torch.load(model_path)
    else:
        device = torch.device('cpu')
        state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    return model, device

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {image_path} - {str(e)}")
        return None

def recognize_single_image(model, device, image_path, class_labels):
    true_class = os.path.basename(os.path.dirname(image_path))
    try:
        true_label = class_labels.index(true_class)
    except ValueError:
        return None, -1, None
    
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return None, true_label, None
    
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_labels[predicted_idx.item()]
    
    is_correct = 1 if predicted_idx.item() == true_label else 0
    return is_correct, true_label, predicted_class

def main():
    model, device = load_model(MODEL_PATH, NUM_CLASSES)
    
    all_images = []
    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                if len(os.path.dirname(img_path).split(os.sep)) > 1:
                    all_images.append(img_path)
    
    if not all_images:
        print(f"No valid images found in {IMAGE_DIR}")
        return
    
    print(f"{len(all_images)} images available for recognition")
    
    start_time = time.time()
    results = []
    processed_count = 0
    
    print("Starting image recognition...")
    try:
        while time.time() - start_time < RUN_DURATION:
            img_path = random.choice(all_images)
            accuracy, true_label, predicted_class = recognize_single_image(
                model, device, img_path, CLASS_LABELS
            )
            
            if accuracy is not None and true_label != -1:
                results.append(accuracy)
                processed_count += 1
                status = "CORRECT" if accuracy == 1 else "INCORRECT"
                print(f"{status} | Image: {os.path.basename(img_path)} | True: {CLASS_LABELS[true_label]} | Predicted: {predicted_class}")
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Process interrupted by user")
    
    if results:
        avg_accuracy = np.mean(results) * 100
        print("\n===== VERIFICATION SUMMARY =====")
        print(f"Total duration: {RUN_DURATION} seconds")
        print(f"Images processed: {processed_count}")
        print(f"Correct predictions: {sum(results)}")
        print(f"Average accuracy: {avg_accuracy:.2f}%")
    else:
        print("No valid images processed")

if __name__ == "__main__":
    main()