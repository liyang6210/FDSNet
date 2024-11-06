import torch
from torchinfo import summary
from torchvision.models.vision_transformer import vit_l_16
from torchvision.models.efficientnet import efficientnet_b7
from src import SegFormer
from src import vit_base_patch16_224_in21k

# 创建模型实例
model = vit_base_patch16_224_in21k()

# 设置模型为评估模式
model.eval()

# 定义输入张量的尺寸

# 使用 torchinfo 获取模型信息
model_info = summary(model, input_size=(1, 3, 512, 512))

# 输出总体信息，包括 MACs
print(model_info)

#
import torch
from thop import profile

model = vit_base_patch16_224_in21k()
input1 = torch.randn(1, 3, 512, 512)
flops, params = profile(model, inputs=(input1, ))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
