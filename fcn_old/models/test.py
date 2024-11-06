import torch
from torchvision.models import swin_transformer
# from src import Model
#
# model = Model()
#
# for k,v in model.state_dict().items():
#     print(k)
print('--------------------------------------------------------------------------------------------------------------------------------')


pth1 = torch.load('D:\\study\\第二个模型试验记录\\版本1\\103\\model_80_46.84.pth')
print(pth1['args'])
# for k,v in pth1['epoch'].items():
#     print(k)




