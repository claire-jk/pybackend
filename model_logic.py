import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# 初始化模型
model = models.mobilenet_v2(pretrained=True)
model.eval()

# 影像預處理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vector(image_data, is_url=False):
    """支援從網址(後台)或二進位流(手機)提取特徵"""
    if is_url:
        response = requests.get(image_data)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(BytesIO(image_data)).convert('RGB')
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        features = model.features(input_batch)
        vector = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return vector.flatten().numpy()