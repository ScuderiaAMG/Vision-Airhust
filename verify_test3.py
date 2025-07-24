import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet152, efficientnet_b7, resnet101
import os
import time
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = '/home/legion/D/checkpoints/model_5h.pt'
IMAGE_DIR = '/home/legion/dataset/hubei'
NUM_CLASSES = 16
CLASS_LABELS = ['camel', 'chair', 'dolphin', 'lion', 'hamster', 'maple', 'orange', 'orchid', 
                'pickuptruck', 'pinetree', 'rabbit', 'skycraper', 'squirrel', 'tractor', 'turtle', 'willow']

def create_model(num_classes, model_type="resnet152"):
    if model_type == "resnet152":
        model = resnet152(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_type == "efficientnet_b7":
        model = efficientnet_b7(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        model = resnet101(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    return model

def load_model(model_path, num_classes, model_type):
    model = create_model(num_classes, model_type)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(model_path))
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
    
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

def predict_image(model, device, image_path, class_labels):
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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_labels[predicted_idx.item()]
    
    is_correct = 1 if predicted_idx.item() == true_label else 0
    return is_correct, true_label, predicted_class, confidence.item()

def main():
    print("=" * 50)
    print("MODEL VERIFICATION SYSTEM")
    print("=" * 50)
    
    model, device = load_model(MODEL_PATH, NUM_CLASSES, "resnet152")
    
    all_images = []
    class_counts = {cls: 0 for cls in CLASS_LABELS}
    class_correct = {cls: 0 for cls in CLASS_LABELS}
    
    for class_name in CLASS_LABELS:
        class_dir = os.path.join(IMAGE_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Missing class directory - {class_dir}")
            continue
            
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, file)
                all_images.append(img_path)
                class_counts[class_name] += 1
    
    if not all_images:
        print(f"No valid images found in {IMAGE_DIR}")
        return
    
    print(f"{len(all_images)} images available for verification")
    print("Starting verification process...")
    
    results = []
    processed_count = 0
    
    start_time = time.time()
    
    for img_path in all_images:
        accuracy, true_label, predicted_class, confidence = predict_image(
            model, device, img_path, CLASS_LABELS
        )
        
        if accuracy is not None and true_label != -1:
            results.append(accuracy)
            processed_count += 1
            
            true_class = CLASS_LABELS[true_label]
            if accuracy == 1:
                class_correct[true_class] += 1
            
            status = "CORRECT" if accuracy == 1 else "INCORRECT"
            print(f"{status} | Image: {os.path.basename(img_path)} | True: {true_class} | Predicted: {predicted_class} | Confidence: {confidence:.2f}")
    
    if results:
        total_time = time.time() - start_time
        avg_accuracy = np.mean(results) * 100
        avg_time_per_image = total_time / len(all_images) * 1000
        
        print("\n" + "=" * 50)
        print("VERIFICATION SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {len(all_images)}")
        print(f"Correct predictions: {sum(results)}")
        print(f"Overall accuracy: {avg_accuracy:.2f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.2f} ms")
        
        print("\nCLASS-WISE ACCURACY:")
        for cls in CLASS_LABELS:
            if class_counts[cls] > 0:
                cls_acc = class_correct[cls] / class_counts[cls] * 100
                print(f"{cls.upper():<15}: {cls_acc:.2f}% ({class_correct[cls]}/{class_counts[cls]})")
    else:
        print("No valid images processed")

if __name__ == "__main__":
    main()