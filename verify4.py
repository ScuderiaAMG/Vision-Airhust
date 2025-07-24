# # import torch
# # import torch.nn as nn
# # import torchvision
# # import numpy as np
# # import cv2
# # import os
# # import time
# # import random
# # from tqdm import tqdm
# # import albumentations as A
# # from albumentations.pytorch import ToTensorV2
# # import psutil
# # import pynvml

# # # 硬件配置
# # pynvml.nvmlInit()
# # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# # gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
# # cpu_cores = psutil.cpu_count(logical=False)
# # logical_cores = psutil.cpu_count(logical=True)

# # config = {
# #     "NUM_WORKERS": min(cpu_cores * 4, logical_cores - 4),
# #     "MODEL": "resnet152",
# #     "GPU_MEMORY": gpu_mem
# # }

# # # 文件路径
# # ORIGINAL_DATASET_ROOT = os.path.expanduser('~/dataset/hubei')
# # FINAL_MODEL_PATH = os.path.expanduser('~/D/data/final_model.pt')

# # class HighSpeedValidator:
# #     def __init__(self):
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #         self.model = self.load_model()
# #         self.transform = self.get_validation_transform()
# #         self.class_map = self.build_class_map()
# #         self.total_samples = self.calculate_total_samples()
        
# #     def load_model(self):
# #         """加载训练好的模型"""
# #         model = create_power_model(10)  # 类别数量需要与实际匹配
# #         state_dict = torch.load(FINAL_MODEL_PATH)
# #         model.load_state_dict(state_dict)
# #         model.to(self.device)
# #         model.eval()
# #         print(f"Model loaded on {self.device}")
# #         return model
    
# #     def get_validation_transform(self):
# #         """验证集专用转换"""
# #         return A.Compose([
# #             A.Resize(256, 256),
# #             A.CenterCrop(224, 224),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ])
    
# #     def build_class_map(self):
# #         """构建类别映射"""
# #         class_map = {}
# #         for class_idx, class_name in enumerate(sorted(os.listdir(ORIGINAL_DATASET_ROOT)):
# #             class_dir = os.path.join(ORIGINAL_DATASET_ROOT, class_name)
# #             class_map[class_name] = {
# #                 'idx': class_idx,
# #                 'samples': [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
# #                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# #             }
# #         return class_map
    
# #     def calculate_total_samples(self):
# #         """计算总样本数"""
# #         return sum(len(v['samples']) for v in self.class_map.values())
    
# #     def get_random_sample(self):
# #         """随机获取一个样本"""
# #         class_name = random.choice(list(self.class_map.keys()))
# #         class_info = self.class_map[class_name]
# #         img_path = random.choice(class_info['samples'])
# #         return img_path, class_info['idx']
    
# #     def preprocess_image(self, img_path):
# #         """预处理图像"""
# #         img = cv2.imread(img_path)
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         img = cv2.resize(img, (256, 256))
# #         transformed = self.transform(image=img)
# #         return transformed["image"]
    
# #     def predict(self, images):
# #         """批量预测"""
# #         with torch.no_grad(), torch.cuda.amp.autocast():
# #             images = images.to(self.device)
# #             outputs = self.model(images)
# #             _, preds = torch.max(outputs, 1)
# #         return preds.cpu()
    
# #     def run_validation(self, duration_hours=1):
# #         """运行验证任务"""
# #         samples_per_second = 8
# #         total_seconds = duration_hours * 3600
# #         total_samples = total_seconds * samples_per_second
# #         batch_size = 32
        
# #         correct = 0
# #         processed = 0
# #         start_time = time.time()
        
# #         print(f"Starting validation for {duration_hours} hour(s)...")
# #         print(f"Target: {total_samples} samples at {samples_per_second} samples/sec")
        
# #         # 预热GPU
# #         warmup_data = torch.rand(16, 3, 224, 224).to(self.device)
# #         for _ in range(10):
# #             _ = self.model(warmup_data)
# #         torch.cuda.synchronize()
        
# #         # 创建进度条
# #         progress = tqdm(total=total_samples, unit="sample", dynamic_ncols=True)
        
# #         while processed < total_samples:
# #             batch_images = []
# #             batch_labels = []
            
# #             # 准备一个batch的数据
# #             for _ in range(min(batch_size, total_samples - processed)):
# #                 img_path, true_label = self.get_random_sample()
# #                 image_tensor = self.preprocess_image(img_path)
# #                 batch_images.append(image_tensor)
# #                 batch_labels.append(true_label)
# #                 processed += 1
# #                 progress.update(1)
            
# #             # 转换为张量并预测
# #             image_batch = torch.stack(batch_images)
# #             true_labels = torch.tensor(batch_labels)
# #             pred_labels = self.predict(image_batch)
            
# #             # 计算准确率
# #             batch_correct = (pred_labels == true_labels).sum().item()
# #             correct += batch_correct
            
# #             # 实时更新准确率
# #             elapsed = time.time() - start_time
# #             current_acc = correct / processed
            
# #             progress.set_postfix({
# #                 "accuracy": f"{current_acc:.4f}",
# #                 "samples/sec": f"{processed/elapsed:.2f}",
# #                 "processed": processed
# #             })
            
# #             # 控制速度
# #             target_time = processed / samples_per_second
# #             if elapsed < target_time:
# #                 time.sleep(max(0, target_time - elapsed - 0.1))
        
# #         progress.close()
# #         final_acc = correct / total_samples
# #         final_speed = total_samples / (time.time() - start_time)
        
# #         print(f"\nValidation completed in {time.time()-start_time:.2f} seconds")
# #         print(f"Final accuracy: {final_acc:.4f}")
# #         print(f"Average speed: {final_speed:.2f} samples/sec")
# #         print(f"Total samples processed: {total_samples}")

# # def create_power_model(num_classes):
# #     """创建模型结构（与训练代码一致）"""
# #     model_name = config['MODEL']
# #     if model_name == "resnet50":
# #         model = torchvision.models.resnet50(weights=None)
# #     elif model_name == "efficientnet_b7":
# #         model = torchvision.models.efficientnet_b7(weights=None)
# #     elif model_name == "resnet152":
# #         model = torchvision.models.resnet152(weights=None)
# #     else:
# #         model = torchvision.models.resnet101(weights=None)
    
# #     if hasattr(model, 'fc'):
# #         num_features = model.fc.in_features
# #         model.fc = nn.Linear(num_features, num_classes)
# #     elif hasattr(model, 'classifier'):
# #         num_features = model.classifier[1].in_features
# #         model.classifier[1] = nn.Linear(num_features, num_classes)
    
# #     return model

# # if __name__ == "__main__":
# #     validator = HighSpeedValidator()
# #     validator.run_validation(duration_hours=1)
# import torch
# import torch.nn as nn
# import torchvision
# import numpy as np
# import cv2
# import os
# import time
# import random
# from tqdm import tqdm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import psutil
# import pynvml

# # 硬件配置
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
# cpu_cores = psutil.cpu_count(logical=False)
# logical_cores = psutil.cpu_count(logical=True)

# config = {
#     "NUM_WORKERS": min(cpu_cores * 4, logical_cores - 4),
#     "MODEL": "resnet152",
#     "GPU_MEMORY": gpu_mem
# }

# # 文件路径
# ORIGINAL_DATASET_ROOT = os.path.expanduser('~/dataset/hubei')
# FINAL_MODEL_PATH = os.path.expanduser('/home/legion/D/checkpoints/model_5h.pt')

# class HighSpeedValidator:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.load_model()
#         self.transform = self.get_validation_transform()
#         self.class_map = self.build_class_map()
#         self.total_samples = self.calculate_total_samples()
        
#     def load_model(self):
#         """加载训练好的模型"""
#         model = create_power_model(10)  # 类别数量需要与实际匹配
#         state_dict = torch.load(FINAL_MODEL_PATH)
#         model.load_state_dict(state_dict)
#         model.to(self.device)
#         model.eval()
#         print(f"Model loaded on {self.device}")
#         return model
    
#     def get_validation_transform(self):
#         """验证集专用转换"""
#         return A.Compose([
#             A.Resize(256, 256),
#             A.CenterCrop(224, 224),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2()
#         ])
    
#     def build_class_map(self):
#         """构建类别映射"""
#         class_map = {}
#         class_names = sorted(os.listdir(ORIGINAL_DATASET_ROOT))
#         for class_idx, class_name in enumerate(class_names):
#             class_dir = os.path.join(ORIGINAL_DATASET_ROOT, class_name)
#             class_map[class_name] = {
#                 'idx': class_idx,
#                 'samples': [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
#                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#             }
#         return class_map
    
#     def calculate_total_samples(self):
#         """计算总样本数"""
#         return sum(len(v['samples']) for v in self.class_map.values())
    
#     def get_random_sample(self):
#         """随机获取一个样本"""
#         class_name = random.choice(list(self.class_map.keys()))
#         class_info = self.class_map[class_name]
#         img_path = random.choice(class_info['samples'])
#         return img_path, class_info['idx']
    
#     def preprocess_image(self, img_path):
#         """预处理图像"""
#         img = cv2.imread(img_path)
#         if img is None:
#             return None
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (256, 256))
#         transformed = self.transform(image=img)
#         return transformed["image"]
    
#     def predict(self, images):
#         """批量预测"""
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             images = images.to(self.device)
#             outputs = self.model(images)
#             _, preds = torch.max(outputs, 1)
#         return preds.cpu()
    
#     def run_validation(self, duration_hours=1):
#         """运行验证任务"""
#         samples_per_second = 8
#         total_seconds = duration_hours * 3600
#         total_samples = total_seconds * samples_per_second
#         batch_size = 32
        
#         correct = 0
#         processed = 0
#         start_time = time.time()
        
#         print(f"Starting validation for {duration_hours} hour(s)...")
#         print(f"Target: {total_samples} samples at {samples_per_second} samples/sec")
        
#         # 预热GPU
#         warmup_data = torch.rand(16, 3, 224, 224).to(self.device)
#         for _ in range(10):
#             _ = self.model(warmup_data)
#         torch.cuda.synchronize()
        
#         # 创建进度条
#         progress = tqdm(total=total_samples, unit="sample", dynamic_ncols=True)
        
#         while processed < total_samples:
#             batch_images = []
#             batch_labels = []
            
#             # 准备一个batch的数据
#             for _ in range(min(batch_size, total_samples - processed)):
#                 img_path, true_label = self.get_random_sample()
#                 image_tensor = self.preprocess_image(img_path)
#                 if image_tensor is None:
#                     continue  # 跳过无效图像
#                 batch_images.append(image_tensor)
#                 batch_labels.append(true_label)
#                 processed += 1
#                 progress.update(1)
            
#             if not batch_images:
#                 continue
                
#             # 转换为张量并预测
#             image_batch = torch.stack(batch_images)
#             true_labels = torch.tensor(batch_labels)
#             pred_labels = self.predict(image_batch)
            
#             # 计算准确率
#             batch_correct = (pred_labels == true_labels).sum().item()
#             correct += batch_correct
            
#             # 实时更新准确率
#             elapsed = time.time() - start_time
#             current_acc = correct / processed if processed > 0 else 0.0
            
#             progress.set_postfix({
#                 "accuracy": f"{current_acc:.4f}",
#                 "samples/sec": f"{processed/elapsed:.2f}" if elapsed > 0 else "0.00",
#                 "processed": processed
#             })
            
#             # 控制速度
#             target_time = processed / samples_per_second
#             if elapsed < target_time:
#                 time.sleep(max(0, target_time - elapsed - 0.1))
        
#         progress.close()
#         final_acc = correct / processed if processed > 0 else 0.0
#         final_speed = processed / (time.time() - start_time) if processed > 0 else 0.0
        
#         print(f"\nValidation completed in {time.time()-start_time:.2f} seconds")
#         print(f"Final accuracy: {final_acc:.4f}")
#         print(f"Average speed: {final_speed:.2f} samples/sec")
#         print(f"Total samples processed: {processed}")

# def create_power_model(num_classes):
#     """创建模型结构（与训练代码一致）"""
#     model_name = config['MODEL']
#     if model_name == "resnet50":
#         model = torchvision.models.resnet50(weights=None)
#     elif model_name == "efficientnet_b7":
#         model = torchvision.models.efficientnet_b7(weights=None)
#     elif model_name == "resnet152":
#         model = torchvision.models.resnet152(weights=None)
#     else:
#         model = torchvision.models.resnet101(weights=None)
    
#     if hasattr(model, 'fc'):
#         num_features = model.fc.in_features
#         model.fc = nn.Linear(num_features, num_classes)
#     elif hasattr(model, 'classifier'):
#         num_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(num_features, num_classes)
    
#     return model

# if __name__ == "__main__":
#     validator = HighSpeedValidator()
#     validator.run_validation(duration_hours=1)
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import os
import time
import random
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import psutil
import pynvml
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=FutureWarning)

# 硬件配置
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
cpu_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)

config = {
    "NUM_WORKERS": min(cpu_cores * 4, logical_cores - 4),
    "MODEL": "resnet152",
    "GPU_MEMORY": gpu_mem
}

# 文件路径
ORIGINAL_DATASET_ROOT = os.path.expanduser('~/dataset/hubei')
FINAL_MODEL_PATH = os.path.expanduser('/home/legion/D/checkpoints/model_5h.pt')

# 定义与训练代码相同的模型包装类
class CheckpointedResNet(nn.Module):
    def __init__(self, model, num_segments=4): 
        super().__init__()
        self.model = model
        self.num_segments = num_segments
    
    def forward(self, x):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        
        # 简化版本，不使用检查点
        x = model.layer3(x)
        x = model.layer4(x)
        
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)
        return x

def create_power_model(num_classes):
    """创建与训练代码相同的模型结构"""
    model_name = config['MODEL']
    if model_name == "resnet50":
        base_model = torchvision.models.resnet50(weights=None)
    elif model_name == "efficientnet_b7":
        base_model = torchvision.models.efficientnet_b7(weights=None)
    elif model_name == "resnet152":
        base_model = torchvision.models.resnet152(weights=None)
    else:
        base_model = torchvision.models.resnet101(weights=None)
    
    if hasattr(base_model, 'fc'):
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(base_model, 'classifier'):
        num_features = base_model.classifier[1].in_features
        base_model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return CheckpointedResNet(base_model, num_segments=4)

class HighSpeedValidator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.transform = self.get_validation_transform()
        self.class_map = self.build_class_map()
        
    def load_model(self):
        """正确加载训练好的模型"""
        model = create_power_model(10)  # 类别数量需要与实际匹配
        
        # 正确加载模型状态
        try:
            # 尝试加载完整检查点
            checkpoint = torch.load(FINAL_MODEL_PATH, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        except RuntimeError:
            # 如果失败，尝试直接加载状态字典
            state_dict = torch.load(FINAL_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        print(f"Model loaded on {self.device}")
        return model
    
    def get_validation_transform(self):
        """验证集专用转换"""
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def build_class_map(self):
        """构建类别映射"""
        class_map = {}
        class_names = sorted(os.listdir(ORIGINAL_DATASET_ROOT))
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(ORIGINAL_DATASET_ROOT, class_name)
            class_map[class_name] = {
                'idx': class_idx,
                'samples': [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            }
        return class_map
    
    def get_random_sample(self):
        """随机获取一个样本"""
        class_name = random.choice(list(self.class_map.keys()))
        class_info = self.class_map[class_name]
        img_path = random.choice(class_info['samples'])
        return img_path, class_info['idx']
    
    def preprocess_image(self, img_path):
        """预处理图像"""
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        transformed = self.transform(image=img)
        return transformed["image"]
    
    def predict(self, images):
        """批量预测"""
        with torch.no_grad(), torch.cuda.amp.autocast():
            images = images.to(self.device)
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)
        return preds.cpu()
    
    def run_validation(self, duration_hours=1):
        """运行验证任务"""
        samples_per_second = 8
        total_seconds = duration_hours * 3600
        total_samples = total_seconds * samples_per_second
        batch_size = 32
        
        correct = 0
        processed = 0
        start_time = time.time()
        
        print(f"Starting validation for {duration_hours} hour(s)...")
        print(f"Target: {total_samples} samples at {samples_per_second} samples/sec")
        
        # 预热GPU
        warmup_data = torch.rand(16, 3, 224, 224).to(self.device)
        for _ in range(10):
            _ = self.model(warmup_data)
        torch.cuda.synchronize()
        
        # 创建进度条
        progress = tqdm(total=total_samples, unit="sample", dynamic_ncols=True)
        
        while processed < total_samples:
            batch_images = []
            batch_labels = []
            
            # 准备一个batch的数据
            for _ in range(min(batch_size, total_samples - processed)):
                img_path, true_label = self.get_random_sample()
                image_tensor = self.preprocess_image(img_path)
                if image_tensor is None:
                    continue  # 跳过无效图像
                batch_images.append(image_tensor)
                batch_labels.append(true_label)
                processed += 1
                progress.update(1)
            
            if not batch_images:
                continue
                
            # 转换为张量并预测
            image_batch = torch.stack(batch_images)
            true_labels = torch.tensor(batch_labels)
            pred_labels = self.predict(image_batch)
            
            # 计算准确率
            batch_correct = (pred_labels == true_labels).sum().item()
            correct += batch_correct
            
            # 实时更新准确率
            elapsed = time.time() - start_time
            current_acc = correct / processed if processed > 0 else 0.0
            
            progress.set_postfix({
                "accuracy": f"{current_acc:.4f}",
                "samples/sec": f"{processed/elapsed:.2f}" if elapsed > 0 else "0.00",
                "processed": processed
            })
            
            # 控制速度
            target_time = processed / samples_per_second
            if elapsed < target_time:
                time.sleep(max(0, target_time - elapsed - 0.1))
        
        progress.close()
        final_acc = correct / processed if processed > 0 else 0.0
        final_speed = processed / (time.time() - start_time) if processed > 0 else 0.0
        
        print(f"\nValidation completed in {time.time()-start_time:.2f} seconds")
        print(f"Final accuracy: {final_acc:.4f}")
        print(f"Average speed: {final_speed:.2f} samples/sec")
        print(f"Total samples processed: {processed}")

if __name__ == "__main__":
    validator = HighSpeedValidator()
    validator.run_validation(duration_hours=1)