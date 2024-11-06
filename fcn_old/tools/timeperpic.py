import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import time
from PIL import Image

# 加载预训练模型
model = resnet50(pretrained=True)
model.eval()  # 设置为评估模式

# 准备数据
transform = transforms.Compose([
    transforms.Resize(768),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载一张图片
image = Image.open('path_to_your_image.jpg')
input_tensor = transform(image).unsqueeze(0)  # 添加批次维度

# 推理并测量时间
start_time = time.time()
with torch.no_grad():  # 确保不会计算梯度
    output = model(input_tensor)
end_time = time.time()

print(f"Time taken to process the image: {end_time - start_time:.3f} seconds")
